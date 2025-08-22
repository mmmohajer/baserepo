import { useState, useEffect, useRef } from "react";

const VoiceRecorder = ({
  onChunk,
  onComplete,
  recording = false,
  setRecording = null,
  chunkDurationInSecond = 15,
}) => {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunksRef.current = [];
    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
        if (onChunk) {
          onChunk(event.data);
        }
      }
    };
    mediaRecorderRef.current.onstop = () => {
      const audioBlob = new Blob(audioChunksRef.current, {
        type: "audio/webm",
      });
      if (onComplete) {
        onComplete(audioBlob);
      }
    };
    mediaRecorderRef.current.start(chunkDurationInSecond * 1000);
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  useEffect(() => {
    if (recording && !mediaRecorderRef.current) {
      startRecording();
    } else if (!recording && mediaRecorderRef.current) {
      stopRecording();
    }
  }, [recording]);

  return <></>;
};

export default VoiceRecorder;
