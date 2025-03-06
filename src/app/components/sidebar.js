"use client";

import Image from "next/image";
import "./sidebar.css";

const SideBar = () => {
  return (
    <div className="flex items-center h-screen min-w-10 flex-col bg-[#121212] md:min-w-64">
      <div className="flex items-center justify-center sidebar mt-64">
        <Image src="/brain.png" alt="brain" width={211} height={239} />
      </div>

      <div className="text-white text-2xl font-bold text-center title flex justify-center items-center">
        Parkinson Disease Detection
      </div>
    </div>
  );
};

export default SideBar;
