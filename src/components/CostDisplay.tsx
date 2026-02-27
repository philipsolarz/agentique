interface Props {
  cost: number;
}

function CostDisplay({ cost }: Props) {
  return (
    <div className="cost-display">
      Cost: ${cost.toFixed(4)}
    </div>
  );
}

export default CostDisplay;
