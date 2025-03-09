// "use client";

// import { useState } from "react";
// import { model, handPD } from "./config";
// import Image from "next/image";

// export default function Home() {
//   const [selectedImage, setSelectedImage] = useState("");
//   const [sample, setSample] = useState("");
//   const [patientResult, setPatientResult] = useState("");

//   const handleSelectChange = (event) => {
//     const selectedTitle = event.target.value;
//     const selectedItem = handPD.find((item) => item.title === selectedTitle);

//     setSelectedImage(selectedItem ? selectedItem.image : "");

//     setTimeout(() => {
//       setSample(selectedItem ? selectedItem.processing : "");
//       setPatientResult(selectedItem ? selectedItem.result : "");
//     }, 1500);
//   };

//   return (
//     <div className="mt-11 main w-full text-white">
//       <div className="text-3xl font-bold text-black">
//         Parkinson&apos;s Disease Detection
//       </div>

//       <div className="imageChoose mt-4">
//         <label
//           htmlFor="HeadlineAct"
//           className="block text-md font-medium text-black title"
//         >
//           Model Selection
//         </label>

//         <select
//           name="HeadlineAct"
//           id="HeadlineAct"
//           className="w-full rounded-lg border border-black bg-white select text-black sm:text-sm focus:ring-0 focus:outline-none"
//           // onChange={handleSelectChange}
//         >
//           <option value="">Please select</option>
//           {model.map((item) => (
//             <option key={item.title} value={item.title} className="text-black">
//               {item.title}
//             </option>
//           ))}
//         </select>
//       </div>


//       <div className="imageChoose mt-4">
//         <label
//           htmlFor="HeadlineAct"
//           className="block text-md font-medium text-black title"
//         >
//           Select Image
//         </label>

//         <select
//           name="HeadlineAct"
//           id="HeadlineAct"
//           className="w-full rounded-lg border border-black bg-white select text-black sm:text-sm focus:ring-0 focus:outline-none"
//           onChange={handleSelectChange}
//         >
//           <option value="">Please select</option>
//           {handPD.map((item) => (
//             <option key={item.title} value={item.title} className="text-black">
//               {item.title}
//             </option>
//           ))}
//         </select>
//       </div>

//       <div className="flex justify-between gap-10">
//         <div className="flex flex-col boxWrapper">
//           <div className="box">Input Image</div>
//           {selectedImage && (
//             <div className="mt-4">
//               <Image
//                 src={selectedImage}
//                 alt="Selected"
//                 width={250}
//                 height={250}
//                 className="object-cover border border-white generatedImage rounded-lg"
//               />
//             </div>
//           )}
//         </div>

//         <div className="flex flex-col boxWrapper">
//           <div className="box">Processed Image</div>
//           {sample && (
//             <div className="mt-4">
//               <Image
//                 src={sample}
//                 alt="sample"
//                 width={350}
//                 height={350}
//                 className="object-cover border border-white generatedImage rounded-lg"
//               />
//             </div>
//           )}
//         </div>
//       </div>

//       <div className="flex flex-col boxWrapper">
//         <div className="box">Result</div>
//         {patientResult && (
//           <div className="result">
//             <div className="text-2xl font-bold text-black">{patientResult}</div>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }


//1 - Parkinson
//0 - Healthy

"use client";

import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
      
      // Log the file being appended to FormData for debugging
      const formData = new FormData();
      formData.append("file", file);
      console.log("Appended FormData:", formData);
    }
  };
  
  const handlePredict = async () => {
    if (!selectedImage) {
      alert("Please select an image first!");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedImage);

    // Log the contents of FormData
    for (let pair of formData.entries()) {
      console.log(pair[0] + ": " + pair[1]);
    }

    try {
      console.log("Sending image to backend...");
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      
      
      const data = await response.json();
      console.log(data);
      setLoading(false);
      setPrediction(data.prediction);

    } catch (error) {
      console.error("Error type:", error.name);
      console.error("Error message:", error.message);
      console.error("Full error:", error);
      setError(`Connection error: ${error.message}`);
    }
};



  return (
    <div className="mt-11 w-full text-white flex flex-col items-center">
      <h1 className="text-3xl font-bold">VGG-19 Image Classification</h1>

      <div className="mt-4 flex flex-col items-center">
        <label className="text-md font-medium">Upload Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="mt-2"
        />
      </div>

      {previewImage && (
        <div className="mt-4 flex flex-col items-center">
          <p className="font-semibold">Input Image</p>
          <Image
            src={previewImage}
            alt="Uploaded"
            width={250}
            height={250}
            className="object-cover border border-white rounded-lg mt-2"
          />
        </div>
      )}

      <button
        onClick={handlePredict}
        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400"
        disabled={loading}
      >
        {loading ? "Processing..." : "Predict"}
      </button>

      {error && (
        <div className="mt-4 text-red-500 font-semibold">Error: {error}</div>
      )}

      {prediction !== null && (
        <div className="mt-4 text-center">
          <p className="font-semibold">Prediction Result</p>
          <p className="text-2xl font-bold">Prediction: {prediction}</p>
        </div>
      )}
    </div>
  );
}
