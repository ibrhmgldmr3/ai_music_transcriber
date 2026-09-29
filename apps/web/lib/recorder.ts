import { encodeWav } from "./wav";

// Copies each block of microphone samples to the page; loaded from a Blob URL so the
// app ships no separate worklet file.
const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("tap", Tap);
`;

/**
 * Records the microphone as WAV. Echo cancellation, noise suppression and automatic
 * gain are turned off: they are made for speech and would smear a guitar's notes.
 */
export class MicRecorder {
  private chunks: Float32Array[] = [];
  private samples = 0;

  private constructor(
    private readonly ctx: AudioContext,
    private readonly stream: MediaStream,
    private readonly nodes: AudioNode[],
  ) {}

  /** Asks for the microphone; `onLevel` gets the input level (RMS, 0-1) per block. */
  static async start(onLevel?: (level: number) => void): Promise<MicRecorder> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    const ctx = new AudioContext();
    try {
      const url = URL.createObjectURL(new Blob([TAP_WORKLET], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
    } catch (err) {
      stream.getTracks().forEach((track) => track.stop());
      void ctx.close();
      throw err;
    }
    const source = ctx.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(ctx, "tap");
    const mute = ctx.createGain(); // keeps the tap in the rendered graph without playing it back
    mute.gain.value = 0;
    source.connect(tap).connect(mute).connect(ctx.destination);
    const recorder = new MicRecorder(ctx, stream, [source, tap, mute]);
    tap.port.onmessage = (event: MessageEvent<Float32Array>) => {
      recorder.chunks.push(event.data);
      recorder.samples += event.data.length;
      if (onLevel) {
        let sum = 0;
        for (const x of event.data) sum += x * x;
        onLevel(Math.sqrt(sum / event.data.length));
      }
    };
    return recorder;
  }

  get seconds(): number {
    return this.samples / this.ctx.sampleRate;
  }

  /** Stops the microphone and returns the recording. */
  async stop(fileName: string): Promise<File> {
    this.stream.getTracks().forEach((track) => track.stop());
    this.nodes.forEach((node) => node.disconnect());
    const sampleRate = this.ctx.sampleRate;
    await this.ctx.close();
    return new File([encodeWav(this.chunks, sampleRate)], fileName, { type: "audio/wav" });
  }
}
