document.addEventListener("DOMContentLoaded", () => {
  // ── Element references ──────────────────────────────────────────────────────
  const promptInput       = document.getElementById("prompt");
  const negPromptInput    = document.getElementById("negPrompt");
  const disableVaeToggle  = document.getElementById("disableVaeToggle");
  const vaeStateBadge     = document.getElementById("vaeStateBadge");
  const stepsInput        = document.getElementById("steps");
  const stepsVal          = document.getElementById("stepsVal");
  const cfgInput          = document.getElementById("cfg");
  const cfgVal            = document.getElementById("cfgVal");
  const resolutionSelect  = document.getElementById("resolution");
  const seedInput         = document.getElementById("seed");
  const btnRandomSeed     = document.getElementById("btnRandomSeed");
  const btnGenerate       = document.getElementById("btnGenerate");
  const btnGenerateText   = document.getElementById("btnGenerateText");
  const modeBadge         = document.getElementById("modeBadge");

  // Model selector
  const btnModelSd15 = document.getElementById("btnModelSd15");
  const btnModelSdxl = document.getElementById("btnModelSdxl");
  const sd15Status   = document.getElementById("sd15Status");
  const sdxlStatus   = document.getElementById("sdxlStatus");
  let currentModel   = "sd15";

  // Reference image
  const refFileInput      = document.getElementById("refFileInput");
  const refDropzone       = document.getElementById("refDropzone");
  const dropzoneContent   = document.getElementById("dropzoneContent");
  const refPreviewThumb   = document.getElementById("refPreviewThumb");
  const btnClearRef       = document.getElementById("btnClearRef");
  const strengthGroup     = document.getElementById("strengthGroup");
  const strengthInput     = document.getElementById("strength");
  const strengthVal       = document.getElementById("strengthVal");
  const facePreserveToggle = document.getElementById("facePreserveToggle");

  // Viewport
  const emptyState       = document.getElementById("emptyState");
  const loadingOverlay   = document.getElementById("loadingOverlay");
  const loadingStatusText = document.getElementById("loadingStatusText");
  const loadingSubtext   = document.getElementById("loadingSubtext");
  const imageViewer      = document.getElementById("imageViewer");
  const resultImage      = document.getElementById("resultImage");
  const metaMode         = document.getElementById("metaMode");
  const metaTime         = document.getElementById("metaTime");
  const metaSeed         = document.getElementById("metaSeed");
  const metaVae          = document.getElementById("metaVae");
  const metaFace         = document.getElementById("metaFace");
  const btnUseAsRef      = document.getElementById("btnUseAsRef");
  const btnDownload      = document.getElementById("btnDownload");
  const btnDeleteImage   = document.getElementById("btnDeleteImage");
  const galleryStrip     = document.getElementById("galleryStrip");
  const btnRefreshGallery = document.getElementById("btnRefreshGallery");
  const gpuName          = document.getElementById("gpuName");
  const vramFreeText     = document.getElementById("vramFreeText");

  let currentRefBase64       = null;
  let activeImageSrc         = null;
  let currentActiveFilename  = null;

  // ── Model switching ─────────────────────────────────────────────────────────
  function setModel(modelId) {
    currentModel = modelId;
    if (modelId === "sdxl") {
      btnModelSdxl.classList.add("active");
      btnModelSd15.classList.remove("active");
      resolutionSelect.innerHTML = `
        <option value="1024x1024" selected>1024 × 1024 (Square · SDXL Native)</option>
        <option value="896x1152">896 × 1152 (Portrait · SDXL)</option>
        <option value="1152x896">1152 × 896 (Landscape · SDXL)</option>
      `;
      // SDXL always needs VAE on for best faces
      disableVaeToggle.checked = false;
      updateVaeBadge();
    } else {
      btnModelSd15.classList.add("active");
      btnModelSdxl.classList.remove("active");
      resolutionSelect.innerHTML = `
        <option value="512x512" selected>512 × 512 (Standard · SD 1.5)</option>
        <option value="512x768">512 × 768 (Portrait · SD 1.5)</option>
        <option value="768x512">768 × 512 (Landscape · SD 1.5)</option>
      `;
    }
    updateLoadingText();
  }

  btnModelSd15.addEventListener("click", () => setModel("sd15"));
  btnModelSdxl.addEventListener("click", () => setModel("sdxl"));

  // ── Sliders ─────────────────────────────────────────────────────────────────
  stepsInput.addEventListener("input", e => stepsVal.textContent = e.target.value);
  cfgInput.addEventListener("input",   e => cfgVal.textContent = parseFloat(e.target.value).toFixed(1));
  strengthInput.addEventListener("input", e => strengthVal.textContent = parseFloat(e.target.value).toFixed(2));

  // ── VAE badge ───────────────────────────────────────────────────────────────
  function updateVaeBadge() {
    if (disableVaeToggle.checked) {
      vaeStateBadge.textContent = "VAE: DISABLED (Direct Latent Preview)";
      vaeStateBadge.className   = "vae-badge vae-disabled";
    } else {
      vaeStateBadge.textContent = "VAE: ENABLED (Full Photorealistic Details)";
      vaeStateBadge.className   = "vae-badge vae-enabled";
    }
  }
  disableVaeToggle.addEventListener("change", updateVaeBadge);

  // ── Random seed ─────────────────────────────────────────────────────────────
  btnRandomSeed.addEventListener("click", () => { seedInput.value = -1; });

  // ── Loading text helper ──────────────────────────────────────────────────────
  function updateLoadingText() {
    const modelName = currentModel === "sdxl" ? "RealVisXL v4.0 (SDXL)" : "Realistic Vision v5.1";
    const mode = currentRefBase64 ? "Editing image" : "Generating";
    loadingStatusText.textContent = `${mode} with ${modelName}…`;
    loadingSubtext.textContent = "Running on NVIDIA RTX 3090 (Offline)";
  }

  // ── Reference image ──────────────────────────────────────────────────────────
  function setReferenceImage(dataUrl) {
    currentRefBase64 = dataUrl;
    refPreviewThumb.src = dataUrl;
    refPreviewThumb.style.display = "block";
    dropzoneContent.style.display = "none";
    btnClearRef.style.display = "block";
    strengthGroup.style.display = "block";

    modeBadge.textContent = "Mode: Image-to-Image (Edit)";
    modeBadge.style.color = "#38bdf8";
    modeBadge.style.backgroundColor = "rgba(6, 182, 212, 0.15)";
    btnGenerateText.textContent = "Edit / Transform Image";

    // Auto-enable VAE for img2img (crucial for face quality)
    disableVaeToggle.checked = false;
    updateVaeBadge();

    // Good defaults for img2img face editing
    cfgInput.value  = 10.0;
    cfgVal.textContent = "10.0";
    stepsInput.value = 35;
    stepsVal.textContent = "35";
    strengthInput.value = 0.50;
    strengthVal.textContent = "0.50";
  }

  function clearReferenceImage() {
    currentRefBase64 = null;
    refFileInput.value = "";
    refPreviewThumb.src = "";
    refPreviewThumb.style.display = "none";
    dropzoneContent.style.display = "flex";
    btnClearRef.style.display = "none";
    strengthGroup.style.display = "none";
    modeBadge.textContent = "Mode: Text-to-Image";
    modeBadge.style.color = "";
    modeBadge.style.backgroundColor = "";
    btnGenerateText.textContent = "Generate Image";

    // Restore txt2img defaults
    cfgInput.value = 7.5;
    cfgVal.textContent = "7.5";
    stepsInput.value = 30;
    stepsVal.textContent = "30";
  }

  btnClearRef.addEventListener("click", e => { e.stopPropagation(); clearReferenceImage(); });

  refDropzone.addEventListener("click", () => refFileInput.click());

  refFileInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = ev => setReferenceImage(ev.target.result);
      reader.readAsDataURL(file);
    }
  });

  refDropzone.addEventListener("dragover", e => {
    e.preventDefault();
    refDropzone.style.borderColor = "#6366f1";
  });
  refDropzone.addEventListener("dragleave", () => { refDropzone.style.borderColor = ""; });
  refDropzone.addEventListener("drop", e => {
    e.preventDefault();
    refDropzone.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = ev => setReferenceImage(ev.target.result);
      reader.readAsDataURL(file);
    }
  });

  btnUseAsRef.addEventListener("click", () => {
    if (activeImageSrc) {
      setReferenceImage(activeImageSrc);
      refDropzone.scrollIntoView({ behavior: "smooth" });
    }
  });

  // ── Delete image ─────────────────────────────────────────────────────────────
  async function deleteImageFile(filename) {
    if (!filename) return;
    const ok = confirm(`Delete "${filename}" permanently?`);
    if (!ok) return;
    try {
      const res = await fetch(`/api/images/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete image");
      if (currentActiveFilename === filename) {
        currentActiveFilename = null;
        activeImageSrc = null;
        imageViewer.style.display = "none";
        emptyState.style.display = "flex";
      }
      await loadGallery();
    } catch (err) {
      alert("Error deleting image: " + err.message);
    }
  }

  btnDeleteImage.addEventListener("click", () => {
    if (currentActiveFilename) deleteImageFile(currentActiveFilename);
  });

  // ── Status polling ───────────────────────────────────────────────────────────
  async function checkStatus() {
    try {
      const res  = await fetch("/api/status");
      const data = await res.json();
      if (data.device) {
        gpuName.textContent = `${data.device} (${data.vram_total || "24GB"})`;
      }
      if (data.vram_free) {
        vramFreeText.textContent = `VRAM Free: ${data.vram_free}`;
      }

      // SD 1.5 status
      if (data.sd15_loaded) {
        sd15Status.textContent = "● Loaded";
        sd15Status.style.color = "#34d399";
      } else if (data.sd15_ready) {
        sd15Status.textContent = "○ Ready (click to load)";
        sd15Status.style.color = "#fbbf24";
      } else {
        sd15Status.textContent = "✕ Model missing";
        sd15Status.style.color = "#f87171";
      }

      // SDXL status
      if (data.sdxl_loaded) {
        sdxlStatus.textContent = "● Loaded";
        sdxlStatus.style.color = "#34d399";
      } else if (data.sdxl_download_complete) {
        sdxlStatus.textContent = "○ Ready (click to load)";
        sdxlStatus.style.color = "#fbbf24";
      } else if (data.sdxl_ready) {
        sdxlStatus.textContent = `⬇ Downloading… ${data.sdxl_download_gb} / 6.6 GB`;
        sdxlStatus.style.color = "#60a5fa";
      } else {
        sdxlStatus.textContent = "✕ Not downloaded";
        sdxlStatus.style.color = "#f87171";
      }
    } catch (err) {
      console.warn("Status check pending...", err);
    }
  }
  checkStatus();
  setInterval(checkStatus, 8000);

  // ── Gallery ──────────────────────────────────────────────────────────────────
  async function loadGallery() {
    try {
      const res  = await fetch("/api/gallery");
      const data = await res.json();
      galleryStrip.innerHTML = "";
      if (data.images && data.images.length > 0) {
        data.images.forEach(imgFile => {
          const item = document.createElement("div");
          item.className = "gallery-item";

          const img = document.createElement("img");
          img.src = `/outputs/${imgFile}`;
          img.className = "gallery-thumb";
          img.title = imgFile;
          img.addEventListener("click", () => {
            currentActiveFilename = imgFile;
            activeImageSrc = `/outputs/${imgFile}`;
            resultImage.src = `/outputs/${imgFile}`;
            btnDownload.href = `/outputs/${imgFile}`;
            btnDownload.download = imgFile;
            emptyState.style.display = "none";
            imageViewer.style.display = "flex";
            metaMode.textContent = imgFile.startsWith("sdxl") ? "[SDXL]" : "[SD 1.5]";
            metaTime.textContent = "📁 Gallery";
            metaSeed.textContent = imgFile;
            metaVae.textContent  = "Offline";
            metaFace.style.display = "none";
          });

          const delBtn = document.createElement("button");
          delBtn.className = "gallery-delete-btn";
          delBtn.title = "Delete";
          delBtn.textContent = "🗑️";
          delBtn.addEventListener("click", e => { e.stopPropagation(); deleteImageFile(imgFile); });

          item.appendChild(img);
          item.appendChild(delBtn);
          galleryStrip.appendChild(item);
        });
      } else {
        const span = document.createElement("span");
        span.className = "gallery-empty-hint";
        span.textContent = "No images generated yet";
        galleryStrip.appendChild(span);
      }
    } catch (err) {
      console.warn("Gallery load error", err);
    }
  }
  loadGallery();
  btnRefreshGallery.addEventListener("click", loadGallery);

  // ── Generate ─────────────────────────────────────────────────────────────────
  btnGenerate.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt.");
      promptInput.focus();
      return;
    }

    const [w, h] = resolutionSelect.value.split("x").map(Number);
    const disableVae   = disableVaeToggle.checked;
    const isImg2Img    = !!currentRefBase64;
    const facePreserve = isImg2Img && facePreserveToggle.checked;

    btnGenerate.disabled = true;
    emptyState.style.display = "none";
    imageViewer.style.display = "none";
    loadingOverlay.style.display = "flex";
    updateLoadingText();

    try {
      const payload = {
        model_type:      currentModel,
        prompt:          prompt,
        negative_prompt: negPromptInput.value.trim() || null,
        steps:           parseInt(stepsInput.value),
        cfg_scale:       parseFloat(cfgInput.value),
        seed:            parseInt(seedInput.value) || -1,
        width:           w,
        height:          h,
        disable_vae:     disableVae,
        image_base64:    currentRefBase64 || null,
        strength:        isImg2Img ? parseFloat(strengthInput.value) : undefined,
        face_preserve:   facePreserve
      };

      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        let errMsg = "Generation failed";
        try {
          const errText = await response.text();
          const errJson = JSON.parse(errText);
          errMsg = errJson.detail || errMsg;
        } catch (_) {
          errMsg = `Server error (${response.status}). Check if model is loaded.`;
        }
        throw new Error(errMsg);
      }

      const result = await response.json();
      currentActiveFilename = result.filename;
      activeImageSrc = result.image_base64;
      resultImage.src = result.image_base64;
      metaMode.textContent = `[${result.model.toUpperCase()}] ${result.mode}`;
      metaTime.textContent = `⚡ ${result.elapsed}s`;
      metaSeed.textContent = `Seed: ${result.seed}`;
      metaVae.textContent  = result.vae_disabled ? "VAE: Disabled" : "VAE: Enabled";
      if (result.face_preserve) {
        metaFace.style.display = "inline-block";
        metaFace.textContent = `👤 Face Preserve (str=${result.strength_used})`;
      } else {
        metaFace.style.display = "none";
      }
      btnDownload.href     = result.image_base64;
      btnDownload.download = result.filename;

      imageViewer.style.display = "flex";
      loadGallery();
    } catch (err) {
      alert("Error: " + err.message);
      emptyState.style.display = "flex";
    } finally {
      loadingOverlay.style.display = "none";
      btnGenerate.disabled = false;
    }
  });
});
