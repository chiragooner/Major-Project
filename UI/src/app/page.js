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

// "use client";

// import { useState } from "react";
// import Image from "next/image";

// export default function Home() {
//   const [selectedImage, setSelectedImage] = useState(null);
//   const [previewImage, setPreviewImage] = useState(null);
//   const [prediction, setPrediction] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleImageUpload = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       setSelectedImage(file);
//       setPreviewImage(URL.createObjectURL(file));
//       setPrediction(null);
//       setError(null);
      
//       // Log the file being appended to FormData for debugging
//       const formData = new FormData();
//       formData.append("file", file);
//       console.log("Appended FormData:", formData);
//     }
//   };
  
//   const handlePredict = async () => {
//     if (!selectedImage) {
//       alert("Please select an image first!");
//       return;
//     }

//     setLoading(true);
//     setError(null);
//     const formData = new FormData();
//     formData.append("file", selectedImage);

//     // Log the contents of FormData
//     for (let pair of formData.entries()) {
//       console.log(pair[0] + ": " + pair[1]);
//     }

//     try {
//       console.log("Sending image to backend...");
//       const response = await fetch("http://127.0.0.1:8000/predict/vgg19", {
//         method: "POST",
//         body: formData,
//       });
      
      
//       const data = await response.json();
//       console.log(data);
//       setLoading(false);
//       setPrediction(data.prediction);

//     } catch (error) {
//       console.error("Error type:", error.name);
//       console.error("Error message:", error.message);
//       console.error("Full error:", error);
//       setError(`Connection error: ${error.message}`);
//     }
// };



//   return (
//     <div className="mt-11 w-full text-white flex flex-col items-center">
//       <h1 className="text-3xl font-bold">VGG-19 Image Classification</h1>

//       <div className="mt-4 flex flex-col items-center">
//         <label className="text-md font-medium">Upload Image</label>
//         <input
//           type="file"
//           accept="image/*"
//           onChange={handleImageUpload}
//           className="mt-2"
//         />
//       </div>

//       {previewImage && (
//         <div className="mt-4 flex flex-col items-center">
//           <p className="font-semibold">Input Image</p>
//           <Image
//             src={previewImage}
//             alt="Uploaded"
//             width={250}
//             height={250}
//             className="object-cover border border-white rounded-lg mt-2"
//           />
//         </div>
//       )}

//       <button
//         onClick={handlePredict}
//         className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-400"
//         disabled={loading}
//       >
//         {loading ? "Processing..." : "Predict"}
//       </button>

//       {error && (
//         <div className="mt-4 text-red-500 font-semibold">Error: {error}</div>
//       )}

//       {prediction !== null && (
//         <div className="mt-4 text-center">
//           <p className="font-semibold">Prediction Result</p>
//           <p className="text-2xl font-bold">Prediction: {prediction}</p>
//         </div>
//       )}
//     </div>
//   );
// }




"use client";

import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("vgg19");
  const [confidenceScore, setConfidenceScore] = useState(null);
  const [probabilities, setProbabilities] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
      setPrediction(null);
      setConfidenceScore(null);
      setProbabilities(null);
      setError(null);
    }
  };
  
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    // Reset prediction when model changes
    setPrediction(null);
    setConfidenceScore(null);
    setProbabilities(null);
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

    try {
      console.log(`Sending image to backend using ${selectedModel} model...`);
      const response = await fetch(`http://127.0.0.1:8000/predict/${selectedModel}`, {
        method: "POST",
        body: formData,
      });

      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response data:", data);
      setLoading(false);
      setPrediction(data.prediction);
      
      // Set confidence score and probabilities if available
      if (data.confidence) {
        setConfidenceScore(data.confidence);
      }
      
      if (data.probabilities) {
        setProbabilities(data.probabilities);
      }
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
      setError(`Error: ${error.message}`);
    }
  };

  return (
    <div className="mt-11 w-full text-white flex flex-col items-center">
      <h1 className="text-3xl font-bold">VGG Image Classification</h1>

      <div className="mt-8 flex flex-col items-center w-full max-w-md">
        <div className="w-full bg-gray-800 p-6 rounded-lg shadow-lg">
          <div className="mb-4">
            <label className="block text-md font-medium mb-2">Select Model</label>
            <select
              value={selectedModel}
              onChange={handleModelChange}
              className="w-full px-4 py-2 bg-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="vgg19">VGG19 (Wave Flipping)</option>
              <option value="vgg16">VGG16 (Spiral No Aug)</option>
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-md font-medium mb-2">Upload Image</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="w-full bg-gray-700 px-4 py-2 rounded-md text-white"
            />
          </div>
          
          <button
            onClick={handlePredict}
            className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition duration-200 disabled:bg-gray-500 disabled:cursor-not-allowed"
            disabled={loading || !selectedImage}
          >
            {loading ? "Processing..." : "Predict"}
          </button>
        </div>

        {previewImage && (
          <div className="mt-6 flex flex-col items-center">
            <p className="font-semibold text-lg mb-2">Input Image</p>
            <div className="border-2 border-gray-600 rounded-lg p-1 bg-gray-800">
              <Image
                src={previewImage}
                alt="Uploaded"
                width={250}
                height={250}
                className="object-cover rounded-lg"
              />
            </div>
          </div>
        )}

        {error && (
          <div className="mt-6 w-full p-4 bg-red-900/50 border border-red-600 rounded-lg">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {prediction !== null && (
          <div className="mt-6 w-full bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-bold mb-4 text-center">Results</h2>
            
            <div className="mb-4 p-4 bg-gray-700 rounded-lg">
              <p className="text-lg mb-1">Model: <span className="font-semibold">{selectedModel.toUpperCase()}</span></p>
              <p className="text-lg mb-1">Prediction: <span className="font-semibold text-2xl">{prediction}</span></p>
              
              {confidenceScore !== null && (
                <p className="text-lg">Confidence: <span className="font-semibold">{confidenceScore}%</span></p>
              )}
            </div>
            
            {probabilities && (
              <div className="mt-4">
                <p className="font-medium mb-2">Class Probabilities:</p>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(probabilities).map(([className, probability]) => (
                    <div key={className} className="flex justify-between p-2 bg-gray-700 rounded">
                      <span>{className}</span>
                      <span className="font-medium">{probability}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}