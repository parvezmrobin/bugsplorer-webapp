<template>
  <div class="container-fluid">
    <div class="row">
      <div class="col pb-2">
        <h1 class="py-3 text-center">
          Bugsplorer Web App
          <span
            v-show="loading"
            class="spinner-border text-primary fw-lighter"
            style="--bs-spinner-border-width: 0.125em"
            role="status"
          >
            <span class="visually-hidden">Loading...</span>
          </span>
        </h1>
      </div>
    </div>
    <div class="row border-top">
      <div class="col col-3">
        <div class="row mt-3">
          <label for="formFile" class="col-auto col-form-label">
            Select a file
          </label>
          <div class="col-auto">
            <input
              id="formFile"
              class="form-control"
              type="file"
              @input="readFileContent"
            />
          </div>
        </div>
      </div>
      <div class="col col-9 pe-0">
        <pre class="position-relative">
          <span
            v-for="(score, i) in defectPossibilities"
            :key="i"
            class="d-block position-absolute w-100 ps-2"
            style="height: 1.5em; left: 0;"
            :style="{
              backgroundColor: getBackgroundColor(score, i),
              top: `${i * 1.5 + 1}em`
            }"
          >{{
              (i + 1).toString(10).padStart(3)
            }}<template v-if="tokenExplanationIndices.includes(i)">
              <span
                v-for="tokenExplanation in tokenExplanations[i]"
                :key="tokenExplanation.start"
                class="position-absolute"
                style="top: 0; height: 1.5em;"
                :data-attn="tokenExplanation.strength"
                :style="{
                  width: `${tokenExplanation.width/2}rem`,
                  backgroundColor: makeTokenColor(tokenExplanation.strength),
                  left: `${tokenExplanation.start/2.1 + 2.75}rem`}"
              />
            </template></span>

          <code
            ref="fileContentEl"
            class="hljs position-absolute w-100"
            :class="{[`language-${language}`]: true, 'border-start': fileContent.length}"
            style="left: 0; top: 0;"
          >{{ fileContent }}</code>

          <template
            v-for="(score, i) in defectPossibilities"
            :key="i"
          >
            <div
              v-if="!loading && isAboveThreshold(score) && (i == 0 || score !== defectPossibilities[i - 1])"
              class="position-absolute btn-explain-wrapper"
              style="right: 16px;"
              :style="{
                top: `${i * 1.5 + .375}em`
              }"
            >
              <button
                type="button"
                class="btn btn-warning position-relative"
                @click="tokenExplanationIndices.includes(i) ? clearExplanation(): showExplanation(i)"
              >{{ tokenExplanationIndices.includes(i) ? "Hide Explanation" : "Explain" }}</button>
            </div>
          </template>

          <span
            v-for="(score, i) in defectPossibilities"
            :key="i"
            class="d-block position-fixed"
            style="width: 5px; right: 16px; border-width: 2px; border-style: solid"
            :style="{
              borderColor: getBackgroundColor(score, i),
              top: `calc(${i / defectPossibilities.length * 100}% + 100px)`,
            }"
          />
        </pre>
      </div>
    </div>
  </div>

  <dialog ref="dialogEl">
    <p>Something went wrong!</p>
    <form method="dialog">
      <button>Okay</button>
    </form>
  </dialog>

  <div
    v-if="tokenExplanationIndices.length > 0"
    class="d-flex w-100 position-fixed"
    style="bottom: 0; left: 0"
  >
    <div
      v-for="i in 100"
      :key="i"
      style="height: 1em; width: 1%"
      :style="{
        backgroundColor: getBrighterColor(colors[i - 1]),
      }"
    ></div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import hljs from "highlight.js";
import axios from "axios";
import createColormap from "colormap";

const colors = createColormap({
  colormap: "jet",
  nshades: 100,
  format: "rgba",
  alpha: 0.5,
});

const fileContentEl = ref<HTMLElement | null>(null);
const fileContent = ref<string>("");
const dialogEl = ref<HTMLDialogElement | null>(null);

const language = ref<string>("");

const defectPossibilities = ref<number[]>([]);
const attentions = ref<number[][]>([]);
const offests = ref<number[][]>([]);
const tokenExplanationIndices = ref<number[]>([]);
type TokenExplanation = {
  start: number;
  width: number;
  strength: number;
};
const tokenExplanations = ref<Record<number, TokenExplanation[]>>([]);

const minDefectPossibility = computed(() =>
  Math.min(...defectPossibilities.value)
);
const maxDefectPossibility = computed(() =>
  Math.max(...defectPossibilities.value)
);
const loading = ref(false);

function resetStates() {
  defectPossibilities.value = [];
  attentions.value = [];
  offests.value = [];
  tokenExplanationIndices.value = [];
  tokenExplanations.value = [];
}

const readFileContent = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file: File = (target.files as FileList)[0];
  language.value = file.name.split(".").pop() as string;
  resetStates();
  const reader = new FileReader();
  reader.readAsText(file);
  reader.onload = () => {
    if (reader.result) {
      fileContent.value = reader.result as string;
    }
  };
};

watch(fileContent, () => {
  updateDefectPossibilities();
  nextTick(() => {
    hljs.highlightElement(fileContentEl.value as HTMLElement);
  });
});

type Response = {
  attention: number[][];
  defect_prob: number[];
  offset: number[][][];
};

async function updateDefectPossibilities() {
  loading.value = true;
  defectPossibilities.value = Array(fileContent.value.split("\n").length).fill(
    0
  );
  attentions.value = [];
  try {
    const response = await axios.post<Response>(
      `http://localhost:5000/api/explore?lang=${language.value}`,
      fileContent.value
    );
    defectPossibilities.value = response.data.defect_prob;
    attentions.value = response.data.attention;
    offests.value = response.data.offset;
  } catch (e) {
    console.error(e);
    dialogEl.value?.showModal();
  } finally {
    loading.value = false;
  }
}

function getBackgroundColor(score: number, i: number) {
  let opacity: number;
  if (tokenExplanationIndices.value.includes(i)) {
    opacity = 0;
  } else {
    opacity =
      (score - minDefectPossibility.value) /
      (maxDefectPossibility.value - minDefectPossibility.value);
  }
  return `rgba(255, 193, 7, ${opacity})`;
}

function isAboveThreshold(score: number) {
  return (
    score >= (minDefectPossibility.value + maxDefectPossibility.value) * 0.5
  );
}

function showExplanation(index: number) {
  tokenExplanationIndices.value = [];
  for (
    let i = index;
    i < defectPossibilities.value.length &&
    defectPossibilities.value[i] === defectPossibilities.value[index];
    i++
  ) {
    tokenExplanationIndices.value.push(i);
  }

  const allSelectedAttention = tokenExplanationIndices.value
    .map((i) => attentions.value[i])
    .flat()
    .filter(Boolean);
  const minTokenAttn = Math.min(...allSelectedAttention);
  const maxTokenAttn = Math.max(...allSelectedAttention);

  tokenExplanations.value = {};
  for (const index of tokenExplanationIndices.value) {
    const lineAttention = attentions.value[index];
    const lineOffset = offests.value[index];

    tokenExplanations.value[index] = Array(lineOffset.length - 1)
      .fill(0)
      .map((_, i) => {
        const start = lineOffset[i][0];
        const end = lineOffset[i][1];
        const width = Math.max(end - start, 0);
        const strength =
          (lineAttention[i] - minTokenAttn) / (maxTokenAttn - minTokenAttn);
        return { start, width, strength };
      });
  }
}

function clearExplanation() {
  tokenExplanationIndices.value = [];
  tokenExplanations.value = [];
}

const adder = 92;

function getBrighterComp(r: number) {
  return Math.min(r + adder, 255);
}

function getBrighterColor(color: [number, number, number, number]) {
  const [r, g, b] = color;
  const lighter = `rgb(${getBrighterComp(r)}, ${getBrighterComp(
    g
  )}, ${getBrighterComp(b)})`;
  return lighter;
}

/**
 * Get color from {@link colors} using {@link strength} as index.
 * Then make the color brighter and return as a string.
 * @param strength
 */
function makeTokenColor(strength: number) {
  if (strength < 0) {
    return "transparent";
  }
  const color = colors[Math.min(Math.floor(strength * 100), 99)];
  const lighter = getBrighterColor(color);
  return lighter;
}
</script>

<style scoped lang="scss">
pre {
  height: calc(100vh - 96px - 1px); // 96px title + 1px border
  margin-bottom: 0;
}

.hljs {
  background: transparent;
  padding-left: 3em;
}

.btn-explain-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;

  &:hover {
    z-index: 3;
  }
}
</style>
