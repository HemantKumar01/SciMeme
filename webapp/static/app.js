const OPENAI_FALLBACK_MODELS = [
  "gpt-5.6-luna",
  "gpt-5.6-terra",
  "gpt-5.6-sol",
  "gpt-5.4-mini",
  "gpt-5-mini",
  "gpt-4.1-mini",
  "gpt-4o-mini",
];

let OPEN_SOURCE_MODELS = [
  { id: "zai.glm-5", label: "GLM 5" },
  { id: "qwen.qwen3-vl-235b-a22b-instruct", label: "Qwen3 VL 235B A22B" },
  { id: "meta.llama3-70b-instruct-v1:0", label: "Llama 3 70B Instruct" },
];

const STAGE_OUTPUT_NAMES = {
  extraction: "Paper extraction",
  innovation: "Innovative reflections",
  concisio: "Concisio",
  template_catalog: "Template catalog",
  final_evaluation: "Final evaluation",
  rendering: "Final rendering metadata",
};

function outputName(stage) {
  if (STAGE_OUTPUT_NAMES[stage]) return STAGE_OUTPUT_NAMES[stage];
  const match = stage.match(/^(selection|generation|evaluation)_(\d+)$/);
  if (!match) return stage;
  const round = Number(match[2]);
  const phase = round === 0 ? "Initial exploration" : `Iteration ${round}`;
  const action = {
    selection: "template selection",
    generation: "generation",
    evaluation: "evaluation",
  }[match[1]];
  return `${phase} · ${action}`;
}

const form = document.querySelector("#pipeline-form");
const paperInput = document.querySelector("#paper-input");
const fileLabel = document.querySelector("#file-label");
const dropzone = document.querySelector("#dropzone");
const apiKey = document.querySelector("#api-key");
const providerInput = document.querySelector("#provider");
const providerTabs = [...document.querySelectorAll(".provider-tab")];
const openSourcePanel = document.querySelector("#open-source-panel");
const openaiPanel = document.querySelector("#openai-panel");
const openSourceStatus = document.querySelector("#open-source-status");
const toggleKey = document.querySelector("#toggle-key");
const loadModelsButton = document.querySelector("#load-models");
const modelNote = document.querySelector("#model-note");
const runButton = document.querySelector("#run-button");
const paperError = document.querySelector("#paper-error");
const keyError = document.querySelector("#key-error");
const formError = document.querySelector("#form-error");
const progress = document.querySelector("#progress");
const progressTrack = document.querySelector(".progress-track");
const progressFill = document.querySelector("#progress-fill");
const progressValue = document.querySelector("#progress-value");
const currentStep = document.querySelector("#current-step");
const runStatus = document.querySelector("#run-status");
const emptyState = document.querySelector("#empty-state");
const memeImage = document.querySelector("#meme-image");
const resultActions = document.querySelector("#result-actions");
const downloadMeme = document.querySelector("#download-meme");
const downloadJson = document.querySelector("#download-json");
const intermediateList = document.querySelector("#intermediate-list");

const modelFields = {
  innovation: document.querySelector("#innovation-model"),
  concisio: document.querySelector("#concisio-model"),
  tsa: document.querySelector("#tsa-model"),
  generation: document.querySelector("#generation-model"),
  critic: document.querySelector("#critic-model"),
};

const defaults = {
  open_source: {
    innovation: "zai.glm-5",
    concisio: "zai.glm-5",
    tsa: "zai.glm-5",
    generation: "zai.glm-5",
    critic: "qwen.qwen3-vl-235b-a22b-instruct",
  },
  openai: {
    innovation: "gpt-5.6-luna",
    concisio: "gpt-4o-mini",
    tsa: "gpt-5.6-luna",
    generation: "gpt-5.6-luna",
    critic: "gpt-5.6-luna",
  },
};

let exportData = null;
let exportStem = "scimemex";
let stageOutputs = {};

function setModels(models, preserve = false) {
  const options = models.map((model) =>
    typeof model === "string" ? { id: model, label: model } : model,
  );
  Object.entries(modelFields).forEach(([stage, select]) => {
    const desired = preserve
      ? select.value
      : defaults[providerInput.value][stage];
    select.replaceChildren(
      ...options.map((model) => new Option(model.label, model.id)),
    );
    select.value = options.some((model) => model.id === desired)
      ? desired
      : options[0].id;
  });
}

setModels(OPEN_SOURCE_MODELS);

function selectProvider(provider) {
  providerInput.value = provider;
  providerTabs.forEach((tab) => {
    const selected = tab.dataset.provider === provider;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  openSourcePanel.hidden = provider !== "open_source";
  openaiPanel.hidden = provider !== "openai";
  keyError.textContent = "";
  if (provider === "open_source") {
    setModels(OPEN_SOURCE_MODELS);
    modelNote.textContent =
      "Open source models are ready. GLM 5 is the default; Qwen3 VL is the default critic.";
  } else {
    setModels(OPENAI_FALLBACK_MODELS);
    modelNote.textContent =
      "Recommended defaults are selected. Verify your key to load models available to your project.";
  }
}

providerTabs.forEach((tab) =>
  tab.addEventListener("click", () => selectProvider(tab.dataset.provider)),
);
providerTabs.forEach((tab, index) => {
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const nextTab =
      providerTabs[
        (index + offset + providerTabs.length) % providerTabs.length
      ];
    selectProvider(nextTab.dataset.provider);
    nextTab.focus();
  });
});

fetch("/api/providers")
  .then((response) => response.json())
  .then((body) => {
    if (!body.open_source?.available) {
      openSourceStatus.textContent =
        "This option is not configured on the server yet.";
    } else {
      OPEN_SOURCE_MODELS = body.open_source.models || OPEN_SOURCE_MODELS;
      openSourceStatus.textContent = `Three Open Source Models are available.`;
      if (providerInput.value === "open_source") setModels(OPEN_SOURCE_MODELS);
    }
  })
  .catch(() => {});

function showFile(file) {
  paperError.textContent = "";
  fileLabel.textContent = file ? file.name : "Choose or drop a PDF";
}

paperInput.addEventListener("change", () => showFile(paperInput.files[0]));

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  paperInput.files = transfer.files;
  showFile(file);
});

toggleKey.addEventListener("click", () => {
  const show = apiKey.type === "password";
  apiKey.type = show ? "text" : "password";
  toggleKey.textContent = show ? "Hide" : "Show";
  toggleKey.setAttribute("aria-pressed", String(show));
  toggleKey.setAttribute("aria-label", show ? "Hide API key" : "Show API key");
});

loadModelsButton.addEventListener("click", async () => {
  keyError.textContent = "";
  if (providerInput.value === "openai" && apiKey.value.trim().length < 12) {
    keyError.textContent = "Enter an API key first.";
    apiKey.focus();
    return;
  }
  loadModelsButton.disabled = true;
  loadModelsButton.textContent = "Loading…";
  try {
    const response = await fetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey.value.trim() }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Could not load models.");
    setModels(body.models, true);
    modelNote.textContent = `${body.models.length} compatible models loaded from this project.`;
  } catch (error) {
    keyError.textContent = error.message;
  } finally {
    loadModelsButton.disabled = false;
    loadModelsButton.textContent = "Verify and load models";
  }
});

function setProgress(percent, label) {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  currentStep.textContent = label;
  progressValue.textContent = `${safePercent}%`;
  progressFill.style.width = `${safePercent}%`;
  progressTrack.setAttribute("aria-valuenow", String(safePercent));
}

function resetResult() {
  exportData = null;
  stageOutputs = {};
  progress.hidden = false;
  resultActions.hidden = true;
  memeImage.hidden = true;
  memeImage.removeAttribute("src");
  emptyState.hidden = false;
  emptyState.textContent = "The meme will appear when the pipeline finishes.";
  runStatus.textContent = "Running";
  setProgress(0, "Starting pipeline");
  renderIntermediateOutputs();
}

function jsonWithoutImage(value) {
  if (Array.isArray(value)) return value.map(jsonWithoutImage);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => key !== "image_data_url")
        .map(([key, child]) => [key, jsonWithoutImage(child)]),
    );
  }
  return value;
}

function renderIntermediateOutputs() {
  const entries = Object.entries(stageOutputs);
  if (!entries.length) {
    intermediateList.replaceChildren(
      Object.assign(document.createElement("p"), {
        textContent: "No intermediate outputs yet.",
      }),
    );
    return;
  }
  const fragment = document.createDocumentFragment();
  entries.forEach(([stage, data]) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const pre = document.createElement("pre");
    summary.textContent = outputName(stage);
    pre.textContent = JSON.stringify(jsonWithoutImage(data), null, 2);
    details.append(summary, pre);
    fragment.append(details);
  });
  intermediateList.replaceChildren(fragment);
}

function handleEvent(payload) {
  if (payload.event === "stage_started") {
    setProgress(payload.progress ?? 0, payload.message || payload.stage);
    return;
  }
  if (payload.event === "stage_completed") {
    stageOutputs[payload.stage] = payload.data;
    setProgress(payload.progress ?? 0, payload.message || payload.stage);
    renderIntermediateOutputs();
    return;
  }
  if (payload.event === "pipeline_completed") {
    exportData = payload.data;
    const filename = payload.data.paper.filename || "paper";
    exportStem =
      filename.replace(/\.pdf$/i, "").replace(/[^a-z0-9_-]+/gi, "-") || "paper";
    const finalMeme = payload.data.steps.final_meme;
    memeImage.src = finalMeme.image_data_url;
    memeImage.alt = `Generated ${finalMeme.template_name} scientific meme`;
    memeImage.hidden = false;
    emptyState.hidden = true;
    resultActions.hidden = false;
    runStatus.textContent = "Complete";
    setProgress(100, "Meme complete");
    return;
  }
  if (payload.event === "pipeline_error") {
    throw new Error(payload.message);
  }
}

async function consumeNdjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) if (line.trim()) handleEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) handleEvent(JSON.parse(buffer));
}

function validateForm() {
  paperError.textContent = "";
  keyError.textContent = "";
  formError.textContent = "";
  const file = paperInput.files[0];
  if (!file) {
    paperError.textContent = "Choose a PDF.";
    return false;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    paperError.textContent = "The selected file must be a PDF.";
    return false;
  }
  if (file.size > 25 * 1024 * 1024) {
    paperError.textContent = "The selected PDF is larger than 25 MB.";
    return false;
  }
  if (providerInput.value === "openai" && apiKey.value.trim().length < 12) {
    keyError.textContent = "Enter your OpenAI API key.";
    return false;
  }
  return true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validateForm()) return;
  resetResult();
  runButton.disabled = true;
  runButton.textContent = "Generating…";
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      body: new FormData(form),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status}).`);
    }
    await consumeNdjson(response);
    if (!exportData)
      throw new Error("The response ended before a meme was generated.");
  } catch (error) {
    runStatus.textContent = "Failed";
    currentStep.textContent = "Pipeline stopped";
    emptyState.hidden = false;
    emptyState.textContent = error.message;
    formError.textContent = error.message;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Generate meme";
  }
});

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

downloadJson.addEventListener("click", () => {
  if (!exportData) return;
  downloadBlob(
    new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    }),
    `${exportStem}-scimemex.json`,
  );
});

downloadMeme.addEventListener("click", async () => {
  if (!exportData) return;
  const response = await fetch(exportData.steps.final_meme.image_data_url);
  downloadBlob(await response.blob(), `${exportStem}-meme.png`);
});
