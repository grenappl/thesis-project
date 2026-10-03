import SongResult from "@/components/SongResult";
import { MOCK_PREDICTION } from "@/lib/mock-data";


interface SongProps {
  params: Promise<{ id: string }>;
}

export default async function SongPage({ params }: SongProps) {
  const resolvedParams = await params;
  const id: string = resolvedParams.id;

  const result = id === '1' ? MOCK_PREDICTION: {
    ...MOCK_PREDICTION,
    songTitle: "Another Wuwa Song"
  }

  return (
    <div className="w-full max-w-xl mx-auto">
      <SongResult result={result} />
    </div>
  )
}
