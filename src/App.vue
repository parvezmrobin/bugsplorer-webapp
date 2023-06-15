<template>
  <div class="container container-fluid">
    <div class="row">
      <div class="col">
        <h1 class="mt-3 mb-5">Bugsplorer Web App</h1>
      </div>
    </div>
    <div class="row mb-3">
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
      <div class="col-auto">
        <div v-show="loading" class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="col">
        <pre class="position-relative">
          <span
            v-for="(score, i) in defectPossibilities"
            :key="i"
            class="d-block position-relative w-100"
            style="height: 1.5em; left: 0; top: -.5em"
            :style="{
              backgroundColor: getBackgroundColor(score),
            }"
          />
          <code
            ref="fileContentEl"
            class="hljs position-absolute w-100"
            :class="`language-${language}`"
            style="left: 0; top: 0;"
          >{{ fileContent }}</code>
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
  defectPossibilities.value = [];
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

<style scoped>
pre {
  overflow: visible;
}

.hljs {
  background: transparent;
  border: 1px solid #ccc;
  border-radius: 4px;
}
</style>
