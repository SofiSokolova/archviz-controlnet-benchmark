const { frames, summary, series, failures } = window.BENCHMARK;

const SCORES = [
  { key: "edge_recall", label: "True boundaries kept", hint: "higher is better", digits: 3 },
  { key: "edge_extra", label: "Lines with no basis in the source", hint: "lower is better", digits: 3 },
  { key: "depth_error", label: "Depth error", hint: "lower is better", digits: 3 },
  { key: "color_shift", label: "Colour shift", hint: "lower is better, about 2 is barely visible", digits: 1 },
  { key: "pixel_change", label: "Changed overall", hint: "how much the model changed at all", digits: 1 },
];
const START_LABELS = { noise: "From noise", render: "From the render" };
const METHOD_LABELS = { canny: "canny", depth: "depth" };
const IMAGE_SIZE = { width: 1216, height: 832 };

function create(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

function createImage(className, loading) {
  return create("img", { class: className, alt: "", ...IMAGE_SIZE, decoding: "async", loading });
}

function createCompare(loading = "eager") {
  const before = createImage("before", loading);
  const after = createImage("after", loading);
  const afterLabel = create("span", { class: "compare-label after-label" });
  const range = create("input", {
    type: "range",
    min: 0,
    max: 100,
    value: 50,
    "aria-label": "Move the divider between the Blender render and the result",
  });
  const root = create(
    "div",
    { class: "compare" },
    before,
    after,
    create("span", { class: "divider" }),
    create("span", { class: "compare-label before", text: "Blender render" }),
    afterLabel,
    range,
  );

  function setPosition(percent) {
    const clamped = Math.min(100, Math.max(0, percent));
    root.style.setProperty("--position", `${clamped}%`);
    range.value = clamped;
  }

  function moveTo(event) {
    const box = root.getBoundingClientRect();
    setPosition(((event.clientX - box.left) / box.width) * 100);
  }

  root.addEventListener("pointerdown", (event) => {
    root.setPointerCapture(event.pointerId);
    moveTo(event);
    range.focus({ preventScroll: true });
  });
  root.addEventListener("pointermove", (event) => {
    if (root.hasPointerCapture(event.pointerId)) moveTo(event);
  });
  range.addEventListener("input", () => setPosition(Number(range.value)));

  return {
    root,
    setPosition,
    show({ source, result, title, label }) {
      before.src = source;
      before.alt = `Blender render of ${title}`;
      after.src = result;
      after.alt = `${label}, ${title}`;
      afterLabel.textContent = label;
    },
  };
}

function createChoice({ legend, name, options, value, variant, onChange }) {
  const list = create("div", { class: "options" });
  for (const option of options) {
    const input = create("input", { type: "radio", name, value: option.value });
    input.checked = option.value === value;
    input.addEventListener("change", () => onChange(option.value));
    list.append(create("label", {}, input, create("span", { class: "option" }, ...option.content)));
  }
  return create("fieldset", { class: `choice ${variant}` }, create("legend", { text: legend }), list);
}

function scoreLabel(score) {
  return create("th", { scope: "row" }, score.label, create("span", { class: "hint", text: score.hint }));
}

function renderCompare(container) {
  const state = { frame: frames[0].id, start: "render", method: "canny" };
  const compare = createCompare();
  const table = create("div", { class: "table-scroll" });

  function update() {
    const frame = frames.find((item) => item.id === state.frame);
    const results = frame.results[state.method];
    compare.show({
      source: frame.source,
      result: results[state.start].image,
      title: frame.title.toLowerCase(),
      label: `${START_LABELS[state.start]}, ${METHOD_LABELS[state.method]}`,
    });

    const columns = [
      { key: "source", title: "Blender render", values: frame.sourceScores },
      { key: "noise", title: "From noise", values: results.noise.scores },
      { key: "render", title: "From the render", values: results.render.scores },
    ];
    const classFor = (key) => (key === state.start ? "is-active" : "");
    table.replaceChildren(
      create(
        "table",
        {},
        create("caption", { text: `Scores for ${frame.title.toLowerCase()}, ${state.method}, seed 1` }),
        create(
          "thead",
          {},
          create(
            "tr",
            {},
            create("th", { scope: "col", text: "Score" }),
            ...columns.map((column) => create("th", { scope: "col", class: classFor(column.key), text: column.title })),
          ),
        ),
        create(
          "tbody",
          {},
          ...SCORES.map((score) =>
            create(
              "tr",
              {},
              scoreLabel(score),
              ...columns.map((column) =>
                create("td", { class: classFor(column.key), text: column.values[score.key].toFixed(score.digits) }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  const change = (key) => (value) => {
    state[key] = value;
    update();
  };

  container.append(
    create(
      "div",
      { class: "controls" },
      createChoice({
        legend: "Frame",
        name: "frame",
        variant: "thumbs",
        value: state.frame,
        onChange: change("frame"),
        options: frames.map((frame) => ({
          value: frame.id,
          content: [create("img", { src: frame.source, alt: "", ...IMAGE_SIZE, loading: "lazy" }), frame.title],
        })),
      }),
      create(
        "div",
        { class: "control-row" },
        createChoice({
          legend: "Start from",
          name: "start",
          variant: "segments",
          value: state.start,
          onChange: change("start"),
          options: [
            { value: "noise", content: ["Noise, the first run"] },
            { value: "render", content: ["The render, denoise 0.4"] },
          ],
        }),
        createChoice({
          legend: "ControlNet input",
          name: "method",
          variant: "segments",
          value: state.method,
          onChange: change("method"),
          options: [
            { value: "canny", content: ["Canny edges"] },
            { value: "depth", content: ["Depth map"] },
          ],
        }),
      ),
    ),
    compare.root,
    table,
  );
  update();
}

function renderFailures(container) {
  for (const failure of failures) {
    const compare = createCompare("lazy");
    compare.show({
      source: failure.source,
      result: failure.result,
      title: failure.title.toLowerCase(),
      label: "Result",
    });
    compare.setPosition(failure.divider);
    container.append(
      create(
        "figure",
        { class: "failure" },
        compare.root,
        create(
          "figcaption",
          {},
          create("h3", { text: failure.title }),
          create("p", { text: failure.text }),
        ),
      ),
    );
  }
}

function renderSeries(container) {
  const state = { denoise: "0.4" };
  const compare = createCompare("lazy");
  const body = create("tbody");

  function update() {
    const step = series.steps.find((item) => item.denoise === state.denoise);
    compare.show({
      source: series.source,
      result: step.image,
      title: "the pavilion terrace",
      label: `Denoise ${step.denoise}`,
    });
    body.replaceChildren(
      ...series.steps.map((item) =>
        create(
          "tr",
          { class: item.denoise === state.denoise ? "is-active" : "" },
          create("th", { scope: "row", text: item.denoise }),
          create("td", { class: "text", text: item.note }),
          create("td", { text: item.floorDeltaE.toFixed(1) }),
          create("td", { text: item.scores.edge_extra.toFixed(3) }),
          create("td", { text: item.scores.pixel_change.toFixed(1) }),
        ),
      ),
    );
  }

  container.append(
    create(
      "div",
      { class: "controls" },
      createChoice({
        legend: "Denoise",
        name: "denoise",
        variant: "segments",
        value: state.denoise,
        onChange: (value) => {
          state.denoise = value;
          update();
        },
        options: series.steps.map((step) => ({ value: step.denoise, content: [step.denoise] })),
      }),
    ),
    compare.root,
    create(
      "div",
      { class: "table-scroll" },
      create(
        "table",
        { class: "series-table" },
        create(
          "thead",
          {},
          create(
            "tr",
            {},
            create("th", { scope: "col", text: "Denoise" }),
            create("th", { scope: "col", class: "text", text: "What I saw" }),
            create("th", { scope: "col", text: "Floor colour difference" }),
            create("th", { scope: "col", text: "Lines with no basis" }),
            create("th", { scope: "col", text: "Changed overall" }),
          ),
        ),
        body,
      ),
    ),
  );
  update();
}

function renderSummary(container) {
  const columns = [
    ["source", "Blender render"],
    ["cannyNoise", "Canny, from noise"],
    ["cannyRender", "Canny, from the render"],
    ["depthNoise", "Depth, from noise"],
    ["depthRender", "Depth, from the render"],
  ];
  container.append(
    create(
      "div",
      { class: "table-scroll" },
      create(
        "table",
        { class: "summary-table" },
        create("caption", { text: "Mean over all images of each method" }),
        create(
          "thead",
          {},
          create(
            "tr",
            {},
            create("th", { scope: "col", text: "Score" }),
            ...columns.map(([, title]) => create("th", { scope: "col", text: title })),
          ),
        ),
        create(
          "tbody",
          {},
          ...SCORES.map((score) =>
            create(
              "tr",
              {},
              scoreLabel(score),
              ...columns.map(([key]) => create("td", { text: summary[score.key][key].toFixed(score.digits) })),
            ),
          ),
        ),
      ),
    ),
  );
}

renderCompare(document.getElementById("compare"));
renderFailures(document.getElementById("failures"));
renderSeries(document.getElementById("series"));
renderSummary(document.getElementById("summary"));
