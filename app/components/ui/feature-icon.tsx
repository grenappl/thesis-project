import { Smile, Activity, Gauge, Zap } from 'lucide-react'

interface FeatureIconProps {
  size: number
}

export default function FeatureIcon({ size }: FeatureIconProps) {
  return (
    <Activity size={size} />
  )
}
