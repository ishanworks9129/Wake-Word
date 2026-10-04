// Copies raw microphone samples to the main thread while recording.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.port.onmessage = (e) => { this.recording = e.data === "start"; };
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (this.recording && channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
