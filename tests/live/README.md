# Live provider acceptance

Set `BATCHWORK_RUN_LIVE=1`, standard provider API-key variables, and one model variable for each flow being tested:

```text
BATCHWORK_LIVE_OPENAI_TEXT_MODEL=gpt-5.5
BATCHWORK_LIVE_OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
BATCHWORK_LIVE_GOOGLE_IMAGES_MODEL=gemini-3-pro-image-preview
BATCHWORK_LIVE_OPENAI_IMAGES_MODEL=gpt-image-2
BATCHWORK_LIVE_XAI_IMAGES_MODEL=grok-imagine-image
```

The naming pattern is `BATCHWORK_LIVE_<PROVIDER>_<MODALITY>_MODEL`. The suite covers seven text, three embedding, and two image flows. Missing model values skip only that flow. These calls create real provider batches and may incur cost.
