"use client";

import { useState } from "react";
import { model, handPD } from "./config";
import Image from "next/image";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState("");
  const [sample, setSample] = useState("");
  const [patientResult, setPatientResult] = useState("");

  const handleSelectChange = (event) => {
    const selectedTitle = event.target.value;
    const selectedItem = handPD.find((item) => item.title === selectedTitle);

    setSelectedImage(selectedItem ? selectedItem.image : "");

    setTimeout(() => {
      setSample(selectedItem ? selectedItem.processing : "");
      setPatientResult(selectedItem ? selectedItem.result : "");
    }, 1500);
  };

  return (
    <div className="mt-11 main w-full text-white">
      <div className="text-3xl font-bold">
        Parkinson&apos;s Disease Detection
      </div>

      <div className="imageChoose mt-4">
        <label
          htmlFor="HeadlineAct"
          className="block text-md font-medium text-white title"
        >
          Model Selection
        </label>

        <select
          name="HeadlineAct"
          id="HeadlineAct"
          className="w-full rounded-lg border border-white bg-white select text-black sm:text-sm focus:ring-0 focus:outline-none"
          // onChange={handleSelectChange}
        >
          <option value="">Please select</option>
          {model.map((item) => (
            <option key={item.title} value={item.title} className="text-black">
              {item.title}
            </option>
          ))}
        </select>
      </div>


      <div className="imageChoose mt-4">
        <label
          htmlFor="HeadlineAct"
          className="block text-md font-medium text-white title"
        >
          Select Image
        </label>

        <select
          name="HeadlineAct"
          id="HeadlineAct"
          className="w-full rounded-lg border border-white bg-white select text-black sm:text-sm focus:ring-0 focus:outline-none"
          onChange={handleSelectChange}
        >
          <option value="">Please select</option>
          {handPD.map((item) => (
            <option key={item.title} value={item.title} className="text-black">
              {item.title}
            </option>
          ))}
        </select>
      </div>

      <div className="flex justify-between gap-10">
        <div className="flex flex-col boxWrapper">
          <div className="box">Input Image</div>
          {selectedImage && (
            <div className="mt-4">
              <Image
                src={selectedImage}
                alt="Selected"
                width={250}
                height={250}
                className="object-cover border border-white generatedImage rounded-lg"
              />
            </div>
          )}
        </div>

        <div className="flex flex-col boxWrapper">
          <div className="box">Processed Image</div>
          {sample && (
            <div className="mt-4">
              <Image
                src={sample}
                alt="sample"
                width={350}
                height={350}
                className="object-cover border border-white generatedImage rounded-lg"
              />
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col boxWrapper">
        <div className="box">Result</div>
        {patientResult && (
          <div className="result">
            <div className="text-2xl font-bold">{patientResult}</div>
          </div>
        )}
      </div>
    </div>
  );
}
