class VoiceCapture extends AudioWorkletProcessor {
  constructor() {super();this.pending=[];this.size=0;this.target=sampleRate;}
  process(inputs) {
    const channel=inputs[0]?.[0];
    if(channel){this.pending.push(new Float32Array(channel));this.size+=channel.length;}
    if(this.size>=this.target){
      const joined=new Float32Array(this.size);let offset=0;
      for(const row of this.pending){joined.set(row,offset);offset+=row.length;}
      const samples=new Float32Array(16000);const scale=sampleRate/16000;
      for(let n=0;n<samples.length;n++){
        const begin=n*scale,end=(n+1)*scale;let total=0,weight=0;
        for(let j=Math.floor(begin);j<Math.ceil(end);j++){
          const amount=Math.max(0,Math.min(j+1,end)-Math.max(j,begin));
          total+=joined[Math.min(j,joined.length-1)]*amount;weight+=amount;
        }
        samples[n]=total/weight;
      }
      this.port.postMessage(samples,[samples.buffer]);
      const remainder=joined.slice(this.target);this.pending=[remainder];this.size=remainder.length;
    }
    return true;
  }
}
registerProcessor('voice-capture',VoiceCapture);
