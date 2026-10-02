// PNG encoding for the model cameras, off the main thread: SensorRig.capture() snapshots the rendered views as
// ImageBitmaps and this worker turns each into a PNG data URL (lossless, so the decoded pixels are the ones rendered).
self.onmessage = async ({ data: { id, bitmaps } }) => {
  try {
    const reader = new FileReaderSync();
    const urls = await Promise.all(
      bitmaps.map(async (bitmap) => {
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
        bitmap.close();
        return reader.readAsDataURL(await canvas.convertToBlob({ type: "image/png" }));
      }),
    );
    self.postMessage({ id, urls });
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
