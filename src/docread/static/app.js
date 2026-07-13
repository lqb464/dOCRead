document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const visionTask = document.getElementById("vision-task");
    const promptWrapper = document.getElementById("prompt-wrapper");
    const promptInput = document.getElementById("prompt-input");
    const processBtn = document.getElementById("process-btn");
    
    const sourceImage = document.getElementById("source-image");
    const canvasPlaceholder = document.getElementById("canvas-placeholder");
    const boxOverlay = document.getElementById("box-overlay");
    const outputBlock = document.getElementById("output-block");
    const engineModeBadge = document.getElementById("engine-mode-badge");

    // Camera Elements
    const cameraBtn = document.getElementById("camera-btn");
    const videoFeed = document.getElementById("video-feed");
    const snapBtn = document.getElementById("snap-btn");
    const cameraCanvas = document.getElementById("camera-canvas");

    let currentFile = null;
    let stream = null;

    // Fetch service health to check mode
    fetch("/api/health")
        .then(res => res.json())
        .then(data => {
            engineModeBadge.textContent = `${data.engine_mode.toUpperCase()} ENGINE`;
            if (data.engine_mode === "cloud" && !data.gemini_api_configured) {
                engineModeBadge.textContent += " (MOCK MODE)";
                engineModeBadge.style.borderColor = "#ef4444";
                engineModeBadge.style.color = "#ef4444";
            }
        })
        .catch(() => {
            engineModeBadge.textContent = "OFFLINE";
            engineModeBadge.style.borderColor = "#ef4444";
        });

    // Mount Static hosting inside server
    // We will mount static files in server.py in Commit 9 as well!

    // Toggle prompt input based on task
    visionTask.addEventListener("change", () => {
        if (visionTask.value === "describe") {
            promptWrapper.style.display = "block";
        } else {
            promptWrapper.style.display = "none";
        }
        clearBoxes();
    });

    // File selection triggers
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    // Drag and drop handlers
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    // Camera handling
    cameraBtn.addEventListener("click", async () => {
        if (stream) {
            stopCamera();
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false
            });
            videoFeed.srcObject = stream;
            videoFeed.style.display = "block";
            snapBtn.style.display = "block";
            cameraBtn.textContent = "⏹️ Stop Camera";
            
            // Hide preview image while streaming
            sourceImage.style.display = "none";
            canvasPlaceholder.style.display = "none";
            clearBoxes();
        } catch (err) {
            alert("Could not access camera: " + err.message);
        }
    });

    snapBtn.addEventListener("click", () => {
        if (!stream) return;

        const ctx = cameraCanvas.getContext("2d");
        cameraCanvas.width = videoFeed.videoWidth;
        cameraCanvas.height = videoFeed.videoHeight;
        
        // Draw mirror image
        ctx.translate(cameraCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        cameraCanvas.toBlob((blob) => {
            const file = new File([blob], "camera_snapshot.jpg", { type: "image/jpeg" });
            handleImageFile(file);
            stopCamera();
        }, "image/jpeg", 0.95);
    });

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        videoFeed.style.display = "none";
        snapBtn.style.display = "none";
        cameraBtn.textContent = "📷 Use Camera";
    }

    function handleImageFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            return;
        }
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            sourceImage.src = e.target.result;
            sourceImage.style.display = "block";
            canvasPlaceholder.style.display = "none";
            processBtn.disabled = false;
            
            // Reset state
            clearBoxes();
            outputBlock.textContent = "Ready to analyze. Click 'Analyze Image' above.";
            outputBlock.classList.add("placeholder-output");
        };
        reader.readAsDataURL(file);
    }

    function clearBoxes() {
        boxOverlay.innerHTML = "";
    }

    // Process analysis
    processBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        processBtn.disabled = true;
        processBtn.textContent = "⌛ Processing...";
        clearBoxes();
        outputBlock.textContent = "Analyzing image. Please wait...";
        outputBlock.classList.add("placeholder-output");

        const task = visionTask.value;
        const formData = new FormData();
        formData.append("file", currentFile);

        let endpoint = `/api/vision/${task}`;
        if (task === "describe" && promptInput.value.strip) {
            formData.append("prompt", promptInput.value.trim());
        }

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Server error occurred");
            }

            const data = await res.json();
            outputBlock.classList.remove("placeholder-output");

            if (task === "ocr") {
                outputBlock.textContent = data.text || "[No text found in image]";
            } else if (task === "describe") {
                outputBlock.textContent = data.description;
            } else if (task === "detect") {
                outputBlock.textContent = JSON.stringify(data.objects, null, 4);
                renderBoundingBoxes(data.objects);
            }
        } catch (err) {
            outputBlock.classList.remove("placeholder-output");
            outputBlock.textContent = `Error: ${err.message}`;
        } finally {
            processBtn.disabled = false;
            processBtn.textContent = "⚡ Analyze Image";
        }
    });

    // Bounding Box Renderer
    function renderBoundingBoxes(objects) {
        clearBoxes();
        
        // Wait for image dimensions to align
        const imgWidth = sourceImage.clientWidth;
        const imgHeight = sourceImage.clientHeight;
        
        boxOverlay.style.width = `${imgWidth}px`;
        boxOverlay.style.height = `${imgHeight}px`;

        objects.forEach(obj => {
            const box = obj.box; // [ymin, xmin, ymax, xmax]
            const ymin = box[0] * imgHeight;
            const xmin = box[1] * imgWidth;
            const ymax = box[2] * imgHeight;
            const xmax = box[3] * imgWidth;
            
            const width = xmax - xmin;
            const height = ymax - ymin;

            const boxEl = document.createElement("div");
            boxEl.className = "bounding-box";
            boxEl.style.top = `${ymin}px`;
            boxEl.style.left = `${xmin}px`;
            boxEl.style.width = `${width}px`;
            boxEl.style.height = `${height}px`;

            const labelEl = document.createElement("span");
            labelEl.className = "box-label";
            labelEl.textContent = `${obj.label} (${Math.round(obj.confidence * 100)}%)`;
            boxEl.appendChild(labelEl);

            boxOverlay.appendChild(boxEl);
        });
    }

    // Handle window resize to keep bounding boxes aligned
    window.addEventListener("resize", () => {
        if (boxOverlay.children.length > 0 && sourceImage.style.display !== "none") {
            const rawJson = outputBlock.textContent;
            try {
                const objects = JSON.parse(rawJson);
                renderBoundingBoxes(objects);
            } catch {
                // ignore if output block is text
            }
        }
    });
});
