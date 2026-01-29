"use client";

import { useState } from "react";

type Result = {
  image_url: string;
  score: number;
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleTextSearch = async () => {
    if (!query) return;

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const res = await fetch(
        `http://localhost:8000/search?query=${encodeURIComponent(query)}`
      );

      if (!res.ok) throw new Error("Search failed");

      const data = await res.json();
      setResults(data.results);
    } catch {
      setError("Failed to fetch text search results");
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async () => {
    if (!imageFile) return;

    setLoading(true);
    setError("");
    setResults([]);

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const res = await fetch(
        "http://localhost:8000/search-by-image?top_k=5",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) throw new Error("Image search failed");

      const data = await res.json();
      setResults(data.results);
    } catch {
      setError("Failed to fetch image search results");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-10 pb-5 bg-black text-white">
      <h1 className="text-5xl font-bold mb-12 py-2 leading-tight bg-linear-to-b from-purple-950 to-cyan-400 bg-clip-text text-transparent">
        Multimodal Image Search
      </h1>

      {/* Text Search */}
      <div className="flex gap-4 mb-6">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search images (e.g. a dog, food, a person)"
          className="border p-2 w-80 rounded"
        />
        <button
          onClick={handleTextSearch}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded border border-blue-400 transition"
        >
          Search
        </button>
      </div>

      {/* Image Search */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-2">
          Search by Image
        </h2>

        <div className="flex items-center gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={(e) =>
              setImageFile(e.target.files?.[0] || null)
            }
            className="h-10 text-sm text-gray-300 file:h-10 file:mr-3 file:px-3 file:rounded file:border-0 file:text-sm file:font-medium file:bg-blue-600 file:text-white hover:file:bg-blue-700 border border-gray-600 rounded focus:outline-none "
          />

          <button
            onClick={handleImageSearch}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded border border-blue-400 transition"
          >
            Search by Image
          </button>
        </div>
        {imageFile && (
          <div className="mt-4">
            <p className="text-sm mb-2 text-gray-400">
              Query Image
            </p>
            <img
              src={URL.createObjectURL(imageFile)}
              alt="query"
              className="w-48 h-48 object-cover rounded border border-gray-600"
            />
          </div>
        )}
      </div>

      {loading && <p className="text-blue-400 animate-pulse">Searching...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {results.length > 0 && (
        <p className="mb-4 text-gray-400">
          Showing top {results.length} results
        </p>
      )}

      {/* Results */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((item, idx) => (
          <div key={idx} className="border border-gray-700 p-2 rounded hover:scale-[1.02] transition">
            <img
              src={item.image_url}
              alt="result"
              className="w-full h-48 object-cover rounded"
            />
            <p className="text-sm mt-2 text-gray-400">
              Score: {item.score.toFixed(3)}
            </p>
          </div>
        ))}
      </div>
    </main>
  );
}
