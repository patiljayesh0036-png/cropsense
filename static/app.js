/* ===== CropSense - front-end logic ===== */
(() => {
  "use strict";

  const MAX_BYTES = 16 * 1024 * 1024;
  const $ = (id) => document.getElementById(id);

  const form = $("analyze-form");
  const hint = $("hint");
  const analyzeBtn = $("analyze-btn");
  const btnText = $("btn-text");
  const errorBox = $("error-box");
  const thermalCol = $("thermal-col");
  const results = $("results");

  const state = {
    crop: null,
    module: null,
    files: { rgb: null, thermal: null },
  };

  const HINTS = {
    none: "Choose a crop and a type of check above, then add your photo here.",
    "rice|water_stress":
      "Add a clear photo of the rice plants. A thermal image is optional and only has a small influence on the result.",
    "rice|nutrients":
      "Add a close photo of a rice leaf. Long, narrow leaf photos are fine. This check looks for nitrogen, phosphorus and potassium deficiency.",
    "maize|water_stress":
      "Add a close-up photo of a single maize leaf. The model learned from small cropped leaf images. Maize does not need a thermal image.",
    "maize|nutrients":
      "Add a close photo of a maize leaf. This check recognises healthy leaves and nitrogen, phosphorus, potassium or magnesium deficiency.",
    partial: "Choose both a crop and a type of check to continue.",
  };

  const LEVEL_TEXT = {
    good: "Looks good",
    warn: "Action needed",
    alert: "Act now",
  };

  const needsThermal = () =>
    state.crop === "rice" && state.module === "water_stress";
  const ready = () => Boolean(state.crop && state.module && state.files.rgb);

  /* ---------- small helpers ---------- */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }
  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  /* ---------- selection ---------- */
  function updateUI() {
    if (state.crop && state.module) {
      hint.textContent = HINTS[`${state.crop}|${state.module}`];
    } else if (state.crop || state.module) {
      hint.textContent = HINTS.partial;
    } else {
      hint.textContent = HINTS.none;
    }

    const showThermal = needsThermal();
    thermalCol.hidden = !showThermal;
    if (!showThermal && state.files.thermal) clearFile("thermal");

    $("card-crop").classList.toggle("is-done", Boolean(state.crop));
    $("card-module").classList.toggle("is-done", Boolean(state.module));
    $("card-upload").classList.toggle("is-done", ready());

    analyzeBtn.disabled = !ready();
  }

  form.addEventListener("change", (e) => {
    if (e.target.name === "crop") state.crop = e.target.value;
    if (e.target.name === "module") state.module = e.target.value;
    if (e.target.name === "crop" || e.target.name === "module") {
      hideError();
      updateUI();
    }
  });

  /* ---------- file upload ---------- */
  function setupDropzone(kind) {
    const zone = $(`drop-${kind}`);
    const input = $(`file-${kind}`);

    zone.addEventListener("click", (e) => {
      if (e.target.closest(".dz-remove")) return;
      input.click();
    });
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) setFile(kind, input.files[0]);
    });

    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
      }),
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
      }),
    );
    zone.addEventListener("drop", (e) => {
      const file =
        e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) setFile(kind, file);
    });
  }

  function setFile(kind, file) {
    hideError();
    if (!file.type.startsWith("image/")) {
      showError("That file is not an image. Choose a JPG, PNG or WEBP photo.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError("That image is larger than 16 MB. Choose a smaller photo.");
      return;
    }
    state.files[kind] = file;

    const zone = $(`drop-${kind}`);
    const img = $(`preview-${kind}`);
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    const url = URL.createObjectURL(file);
    img.src = url;
    img.dataset.url = url;
    $(`name-${kind}`).textContent = file.name;
    zone.querySelector(".dz-empty").hidden = true;
    zone.querySelector(".dz-preview").hidden = false;
    updateUI();
  }

  function clearFile(kind) {
    state.files[kind] = null;
    const zone = $(`drop-${kind}`);
    const img = $(`preview-${kind}`);
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.removeAttribute("src");
    delete img.dataset.url;
    $(`file-${kind}`).value = "";
    zone.querySelector(".dz-empty").hidden = false;
    zone.querySelector(".dz-preview").hidden = true;
    updateUI();
  }

  setupDropzone("rgb");
  setupDropzone("thermal");
  document.querySelectorAll(".dz-remove").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile(btn.dataset.kind);
    }),
  );

  /* ---------- submit ---------- */
  function setLoading(isLoading) {
    analyzeBtn.classList.toggle("is-loading", isLoading);
    analyzeBtn.disabled = isLoading || !ready();
    btnText.textContent = isLoading ? "Analysing..." : "Analyse";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!ready()) return;

    hideError();
    setLoading(true);

    const data = new FormData();
    data.append("crop", state.crop);
    data.append("module", state.module);
    data.append("image", state.files.rgb);
    if (needsThermal() && state.files.thermal)
      data.append("thermal_image", state.files.thermal);

    try {
      const response = await fetch("/predict", { method: "POST", body: data });
      let payload = null;
      try {
        payload = await response.json();
      } catch (_) {
        /* not JSON */
      }
      if (!response.ok || !payload || !payload.ok) {
        throw new Error(
          (payload && payload.error) ||
            "The server returned an error. Try again in a moment.",
        );
      }
      renderResult(payload);
    } catch (err) {
      if (err instanceof TypeError) {
        showError(
          "Could not reach the server. Check that the app is still running in the terminal.",
        );
      } else {
        showError(err.message);
      }
    } finally {
      setLoading(false);
    }
  });

  /* ---------- render result ---------- */
  function renderResult(data) {
    const rec = data.recommendation;
    const level = rec && rec.level ? rec.level : "warn";

    results.classList.remove("level-good", "level-warn", "level-alert");
    results.classList.add(`level-${level}`);

    $("result-badge").textContent = LEVEL_TEXT[level] || "Result";
    $("result-lowconf").hidden = !data.low_confidence;
    $("result-title").textContent = rec ? rec.title : data.prediction;
    $("result-sub").textContent = rec ? rec.summary : "";
    $("result-confidence").textContent = pct(data.confidence);

    // confidence dial: start at 0, then animate to the value
    const gauge = $("result-gauge");
    const value = Math.round(data.confidence * 100);
    gauge.style.setProperty("--pct", "0");
    gauge.setAttribute(
      "aria-label",
      `Confidence of the result: ${value} percent`,
    );
    results.hidden = false;
    requestAnimationFrame(() =>
      requestAnimationFrame(() =>
        gauge.style.setProperty("--pct", String(value)),
      ),
    );

    // uploaded images
    const imageBox = $("result-images");
    imageBox.replaceChildren();
    const captions = { image: "Your photo", thermal_image: "Thermal image" };
    Object.entries(data.images || {}).forEach(([key, url]) => {
      const fig = el("figure");
      const img = el("img");
      img.src = url;
      img.alt = captions[key] || "Uploaded image";
      fig.append(img, el("figcaption", "", captions[key] || ""));
      imageBox.append(fig);
    });

    // probabilities
    const list = $("prob-list");
    list.replaceChildren();
    data.probabilities.forEach((row, i) => {
      const li = el("li", `prob-row${i === 0 ? " top" : ""}`);
      const bar = el("div", "prob-bar");
      const fill = el("div", "prob-fill");
      bar.append(fill);
      li.append(
        el("span", "name", row.label),
        el("span", "pct", pct(row.prob, 1)),
        bar,
      );
      list.append(li);
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          fill.style.width = pct(row.prob, 1);
        }),
      );
    });

    // recommendation
    const recBlock = $("rec-block");
    recBlock.hidden = !rec;
    if (rec) {
      $("rec-title").textContent = "What to do";
      const actions = $("rec-actions");
      actions.replaceChildren();
      rec.actions.forEach((text) => actions.append(el("li", "", text)));
    }

    // warnings
    const warnBlock = $("warnings-block");
    const warnList = $("warnings");
    warnList.replaceChildren();
    (data.warnings || []).forEach((text) =>
      warnList.append(el("li", "", text)),
    );
    warnBlock.hidden = !(data.warnings && data.warnings.length);

    renderDetails(data.details || {});
    $("disclaimer").textContent = data.disclaimer || "";

    const reduce = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    results.scrollIntoView({
      behavior: reduce ? "auto" : "smooth",
      block: "start",
    });
    $("result-title").focus({ preventScroll: true });
  }

  function renderDetails(details) {
    const box = $("tech-details");
    const body = $("details-body");
    body.replaceChildren();

    const rows = [
      ["Photo: deep network", details.rgb_cnn],
      ["Photo: colour model", details.rgb_colour_model],
      ["Photo: combined", details.rgb_combined],
      ["Thermal model", details.thermal],
    ].filter(([, probs]) => probs);

    if (rows.length) {
      const classes = Object.keys(rows[0][1]);
      const table = el("table", "tech-table");
      const head = el("tr");
      head.append(el("th", "", "Model"));
      classes.forEach((c) => head.append(el("th", "", c)));
      table.append(head);
      rows.forEach(([name, probs]) => {
        const tr = el("tr");
        tr.append(el("td", "", name));
        classes.forEach((c) => tr.append(el("td", "", pct(probs[c] ?? 0, 1))));
        table.append(tr);
      });
      body.append(table);
    }
    if (details.rgb_cnn_weight !== undefined) {
      body.append(
        el(
          "p",
          "tech-note",
          `The deep network and the colour model are averaged (deep network share ${pct(details.rgb_cnn_weight)}).`,
        ),
      );
    }
    if (details.fusion_weights) {
      body.append(
        el(
          "p",
          "tech-note",
          `Final answer: photo ${pct(details.fusion_weights.rgb)} and thermal ${pct(details.fusion_weights.thermal)}.`,
        ),
      );
    }
    if (details.tiles_used !== undefined) {
      body.append(
        el(
          "p",
          "tech-note",
          `Your photo was analysed in ${details.tiles_used} piece${details.tiles_used === 1 ? "" : "s"} and the results were averaged.`,
        ),
      );
    }
    box.hidden = body.children.length === 0;
    box.open = false;
  }

  /* ---------- reset and print ---------- */
  $("reset-btn").addEventListener("click", () => {
    results.hidden = true;
    clearFile("rgb");
    clearFile("thermal");
    hideError();
    $("analyse").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("print-btn").addEventListener("click", () => window.print());

  updateUI();
})();
