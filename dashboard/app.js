const D = window.DASHBOARD_DATA;
let direction = "HUST -> MATR";
let cpDirection = "HUST -> MATR";
let tourTimer = null;

const fmt = (value, digits = 0) => Number(value).toFixed(digits);
const metric = (label, value, note, tone = "") =>
  `<div class="metric ${tone}"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`;

function showView(id) {
  document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === id));
  document.querySelectorAll(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === id));
}

function setupNavigation() {
  document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
  document.querySelectorAll(".segmented").forEach(control => {
    control.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
      control.querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
      if (control.dataset.control === "direction") { direction = button.dataset.value; renderTransfer(); }
      if (control.dataset.control === "cpDirection") { cpDirection = button.dataset.value; renderUncertainty(); }
    }));
  });
}

function renderOverview() {
  const s = D.summary;
  document.querySelector("#overviewMetrics").innerHTML = [
    metric("Observed EOL cells", s.observedEol, `${s.totalCells} total cells`, "green"),
    metric("Capacity features", s.features, "First 100 cycles"),
    metric("Shifted feature slopes", `${s.shiftedSlopes} / ${s.features}`, "FDR-corrected", "orange"),
    metric("HUST / MATR life ratio", `${fmt(s.lifeRatio, 2)}×`, "Central lifetime scale", "red"),
  ].join("");
  document.querySelector("#datasetComparison").innerHTML = ["MATR", "HUST"].map(name => {
    const x = D.datasets[name];
    return `<div class="dataset-block ${name.toLowerCase()}"><strong>${name}</strong>
      <div class="dataset-row"><span>Total cells</span><b>${x.n_cells}</b></div>
      <div class="dataset-row"><span>Observed EOL</span><b>${x.n_events}</b></div>
      <div class="dataset-row"><span>Censored</span><b>${x.n_censored}</b></div>
      <div class="dataset-row"><span>Median EOL</span><b>${fmt(x.event_median_cycles)} cycles</b></div>
    </div>`;
  }).join("");
}

function rowsForDirection() {
  return D.point.filter(row => row.Direction === direction);
}

function methodName(setting) {
  return setting
    .replace(", all features", "")
    .replace("Best top-k raw transfer", "Top-k transfer")
    .replace("Target calibration", "Target calibration")
    .replace("CORAL + target calibration", "CORAL + calibration");
}

function barChart(rows, field, higherIsBetter) {
  const values = rows.map(row => Number(row[field]));
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = Math.max(max - min, 1);
  return rows.map(row => {
    const value = Number(row[field]);
    const width = Math.max(3, ((value - min) / span) * 100);
    const setting = row.Setting;
    const tone = setting.startsWith("Target calibration") ? "good" : setting === "CORAL-only" ? "control" : setting.startsWith("Raw") ? "bad" : "";
    return `<div class="bar-row"><span class="bar-label">${methodName(setting)}</span><div class="bar-track"><div class="bar-fill ${tone}" style="width:${width}%"></div></div><span class="bar-value">${fmt(value, field === "R2" ? 3 : 1)}</span></div>`;
  }).join("");
}

function renderTransfer() {
  const rows = rowsForDirection();
  const raw = rows.find(row => row.Setting.startsWith("Raw transfer"));
  const coral = rows.find(row => row.Setting === "CORAL-only");
  const calibrated = rows.find(row => row.Setting.startsWith("Target calibration"));
  const gain = raw.MAE - calibrated.MAE;
  document.querySelector("#transferMetrics").innerHTML = [
    metric("Raw transfer R²", fmt(raw.R2, 3), "Worse than target mean", "red"),
    metric("CORAL-only R²", fmt(coral.R2, 3), "Alignment remains insufficient", "orange"),
    metric("Calibrated R²", fmt(calibrated.R2, 3), "20 labeled target cells", "green"),
    metric("MAE reduction", `${fmt(gain, 1)} cycles`, `${fmt((gain / raw.MAE) * 100, 0)}% vs raw`, "green"),
  ].join("");
  document.querySelector("#r2Chart").innerHTML = barChart(rows, "R2", true);
  document.querySelector("#maeChart").innerHTML = barChart(rows, "MAE", false);
  document.querySelector("#transferInterpretation").innerHTML =
    `<span>Interpretation for ${direction}</span><strong>Feature alignment alone does not restore transfer.</strong><p>Target-side calibration produces the largest improvement, but calibrated R² remains near zero and should be interpreted as error correction rather than strong cross-domain prediction.</p>`;
}

function renderUncertainty() {
  const priority = ["Source CP", "CORAL-after-source CP", "Target-domain CP", "Target-adapted CP"];
  const rows = priority.map(setting => D.conformal.find(row => row.Direction === cpDirection && row.Setting === setting && (row.Model === "catboost" || row.Model === "coral_mlp")));
  const source = rows[0], adapted = rows[3];
  document.querySelector("#cpMetrics").innerHTML = [
    metric("Nominal coverage", "90%", "Evaluation target"),
    metric("Source CP coverage", `${fmt(source.Coverage * 100, 1)}%`, "Under shift", "red"),
    metric("Target-adapted coverage", `${fmt(adapted.Coverage * 100, 1)}%`, "20 target cells", "green"),
    metric("Adapted median width", `${fmt(adapted["Median width"])} cycles`, "Finite intervals", "green"),
  ].join("");
  document.querySelector("#coverageChart").innerHTML = rows.map(row => {
    const good = row.Coverage >= .88 && row.Coverage <= .93;
    const control = row.Setting.startsWith("CORAL");
    return `<div class="coverage-column"><div class="coverage-bar-area"><div class="coverage-bar ${good ? "good" : control ? "control" : ""}" style="height:${Math.min(100, row.Coverage * 100)}%"></div></div><strong>${row.Setting}</strong><span>${fmt(row.Coverage * 100, 1)}% · width ${fmt(row["Median width"])}</span></div>`;
  }).join("");
  const iw = D.iwcp.find(row => row.direction === cpDirection && row.protocol.startsWith("Importance"));
  document.querySelector("#iwFinding").innerHTML =
    `<span>Why importance weighting is insufficient</span><strong>Apparent ${fmt(iw.coverage_mean * 100, 1)}% coverage is obtained with only ${fmt(iw.finite_interval_fraction_mean * 100, 1)}% finite intervals.</strong><p>Dataset discriminator AUC is ${fmt(iw.discriminator_auc_mean, 3)}, indicating weak source–target overlap and unstable density-ratio weighting.</p>`;
}

function pearson(points, feature, dataset) {
  const rows = points.filter(row => row.dataset === dataset);
  const xs = rows.map(row => row[feature]), ys = rows.map(row => row.cycle_life);
  const mx = xs.reduce((a,b) => a+b,0)/xs.length, my = ys.reduce((a,b) => a+b,0)/ys.length;
  const num = xs.reduce((s,x,i) => s+(x-mx)*(ys[i]-my),0);
  const den = Math.sqrt(xs.reduce((s,x) => s+(x-mx)**2,0)*ys.reduce((s,y) => s+(y-my)**2,0));
  return num / den;
}

function scatterSvg(feature) {
  const points = D.featurePoints;
  const width = 1000, height = 410, pad = {l:70,r:25,t:20,b:55};
  const xs = points.map(p => p[feature]), ys = points.map(p => p.cycle_life);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = x => pad.l + ((x-xmin)/(xmax-xmin || 1))*(width-pad.l-pad.r);
  const sy = y => height-pad.b-((y-ymin)/(ymax-ymin || 1))*(height-pad.t-pad.b);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${feature} versus cycle life">`;
  for (let i=0;i<=5;i++) {
    const x=pad.l+i*(width-pad.l-pad.r)/5, y=pad.t+i*(height-pad.t-pad.b)/5;
    svg += `<line class="grid-line" x1="${x}" y1="${pad.t}" x2="${x}" y2="${height-pad.b}"/><line class="grid-line" x1="${pad.l}" y1="${y}" x2="${width-pad.r}" y2="${y}"/>`;
  }
  svg += `<line class="axis" x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}"/>`;
  points.forEach(p => svg += `<circle class="point-${p.dataset.toLowerCase()}" cx="${sx(p[feature])}" cy="${sy(p.cycle_life)}" r="3.5"><title>${p.dataset} ${p.cell_id}: ${feature}=${p[feature].toPrecision(4)}, life=${p.cycle_life}</title></circle>`);
  svg += `<text class="axis-label" x="${width/2}" y="${height-12}" text-anchor="middle">${feature}</text><text class="axis-label" transform="translate(17 ${height/2}) rotate(-90)" text-anchor="middle">Cycle life</text></svg>`;
  return svg;
}

function renderFeatures() {
  const select = document.querySelector("#featureSelect");
  if (!select.options.length) {
    D.features.forEach(feature => select.add(new Option(feature, feature)));
    select.value = "retention_ratio";
    select.addEventListener("change", renderFeatures);
  }
  const feature = select.value;
  const matrR = pearson(D.featurePoints, feature, "MATR");
  const hustR = pearson(D.featurePoints, feature, "HUST");
  document.querySelector("#featureMetrics").innerHTML = [
    metric("MATR correlation", fmt(matrR, 3), "Feature vs cycle life"),
    metric("HUST correlation", fmt(hustR, 3), "Feature vs cycle life"),
    metric("Correlation change", fmt(Math.abs(matrR-hustR), 3), "Relationship instability", "orange"),
    metric("Shifted slopes", `${D.summary.shiftedSlopes} / ${D.summary.features}`, "Across all features", "red"),
  ].join("");
  document.querySelector("#featureChartTitle").textContent = `${feature} vs cycle life`;
  document.querySelector("#scatterChart").innerHTML = scatterSvg(feature);
}

function renderPipeline() {
  document.querySelector("#pipelineList").innerHTML = D.pipeline.map(row =>
    `<div class="pipeline-row"><i></i><strong>${row[0]}</strong><span>${row[1]}</span></div>`
  ).join("");
}

function setupTour() {
  const button = document.querySelector("#tourButton");
  const note = document.querySelector("#tourNote");
  const steps = [
    ["overview", "Start with the research question and the evidence chain."],
    ["transfer", "Show that raw transfer and CORAL remain negative, while target calibration reduces error."],
    ["uncertainty", "Compare source CP with target-adapted CP and explain practical interval validity."],
    ["features", "Use the scatter plot to explain why the same feature can behave differently across domains."],
    ["pipeline", "Close with reproducibility and the final deployment conclusion."],
  ];
  button.addEventListener("click", () => {
    if (tourTimer) {
      clearInterval(tourTimer); tourTimer = null; note.hidden = true; button.textContent = "Start guided demo"; return;
    }
    let i = 0;
    const advance = () => { showView(steps[i][0]); note.textContent = steps[i][1]; note.hidden = false; i = (i+1)%steps.length; };
    advance();
    tourTimer = setInterval(advance, 7000);
    button.textContent = "Stop guided demo";
  });
}

setupNavigation();
renderOverview();
renderTransfer();
renderUncertainty();
renderFeatures();
renderPipeline();
setupTour();
