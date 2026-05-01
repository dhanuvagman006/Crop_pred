import React from 'react'

type CropRow = {
  crop: string
  season: string
  yield: string
  significance: string
}

const CROPS: CropRow[] = [
  {
    crop: 'Rice',
    season: 'Kharif',
    yield: '2,400 – 3,200',
    significance: 'Primary food crop; major cultivation area',
  },
  {
    crop: 'Coconut',
    season: 'Annual',
    yield: '8,500 – 10,000',
    significance: 'Leading plantation crop of coastal Karnataka',
  },
  {
    crop: 'Arecanut',
    season: 'Annual',
    yield: '1,800 – 2,400',
    significance: 'High-value commercial crop; DK specialty',
  },
  {
    crop: 'Banana',
    season: 'Annual',
    yield: '19,000 – 25,000',
    significance: 'Important fruit crop; contributes to horticulture',
  },
  {
    crop: 'Black Pepper',
    season: 'Annual',
    yield: '280 – 420',
    significance: 'Traditional spice crop; high economic value',
  },
  {
    crop: 'Cashewnut',
    season: 'Annual',
    yield: '800 – 1,200',
    significance: 'Plantation crop; important for coastal livelihoods',
  },
  {
    crop: 'Cocoa',
    season: 'Annual',
    yield: '800 – 1,200',
    significance: 'Estate crop; shade-tolerant intercropping',
  },
  {
    crop: 'Sweet Potato',
    season: 'Rabi',
    yield: '15,000 – 25,000',
    significance: 'Root crop; emerging food and processing demand',
  },
]

type Props = {
  className?: string
}

export const CropsTable: React.FC<Props> = ({ className = '' }) => {
  return (
    <section className={className} aria-labelledby="crops-covered-title">
      <h3 id="crops-covered-title" className="text-lg font-semibold mb-3">3.2 Crops Covered</h3>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden">
          <thead className="bg-green-800">
            <tr>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Crop</th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Season</th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Typical Yield Range (kg/ha)</th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">Significance in DK</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {CROPS.map((row, idx) => (
              <tr key={row.crop} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-4 py-3 text-sm text-gray-700 align-top">{row.crop}</td>
                <td className="px-4 py-3 text-sm text-gray-700 align-top">{row.season}</td>
                <td className="px-4 py-3 text-sm text-gray-700 align-top">{row.yield}</td>
                <td className="px-4 py-3 text-sm text-gray-700 align-top">{row.significance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default CropsTable
