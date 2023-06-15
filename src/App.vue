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
    </div>
    <div class="row">
      <div class="col">
        <pre><code ref="fileContentEl" :class="`language-${language}`">{{ fileContent }}</code> </pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import hljs from "highlight.js";

const fileContentEl = ref<HTMLElement | null>(null);
const fileContent = ref<string>("");

const language = ref<string>("");

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

watch(fileContent,async () => {
  await nextTick();
  hljs.highlightElement(fileContentEl.value as HTMLElement);
});
</script>

<style scoped></style>
