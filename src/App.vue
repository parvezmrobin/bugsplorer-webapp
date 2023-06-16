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
              backgroundColor: getBackgroundColor(score),
              top: `${i * 1.5 + 1}em`
            }"
          >{{(i+1).toString(10).padStart(3)}}</span>
          <code
            ref="fileContentEl"
            class="hljs position-absolute w-100"
            :class="{[`language-${language}`]: true, 'border-start': fileContent.length}"
            style="left: 0; top: 0;"
          >{{ fileContent }}</code>

          <span
            v-for="(score, i) in defectPossibilities"
            :key="i"
            class="d-block position-fixed"
            style="width: 5px; right: 16px; border-width: 2px; border-style: solid"
            :style="{
              borderColor: getBackgroundColor(score),
              top: `calc(${i / defectPossibilities.length * 100}% + 100px)`,
            }"
          />
        </pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import hljs from "highlight.js";
import axios from "axios";

const fileContentEl = ref<HTMLElement | null>(null);
const fileContent = ref<string>("");

const language = ref<string>("");

const defectPossibilities = ref<number[]>([]);
const minDefectPossibility = computed(() =>
  Math.min(...defectPossibilities.value)
);
const maxDefectPossibility = computed(() =>
  Math.max(...defectPossibilities.value)
);
const loading = ref(false);

const readFileContent = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file: File = (target.files as FileList)[0];
  language.value = file.name.split(".").pop() as string;
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

async function updateDefectPossibilities() {
  loading.value = true;
  defectPossibilities.value = Array(fileContent.value.split("\n").length).fill(
    0
  );
  const response = await axios.post(
    "http://localhost:5000/api/explore",
    fileContent.value
  );
  defectPossibilities.value = response.data;
  loading.value = false;
}

function getBackgroundColor(score: number) {
  return `rgba(255, 65, 90, ${
    (score - minDefectPossibility.value) /
    (maxDefectPossibility.value - minDefectPossibility.value)
  })`;
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
</style>
