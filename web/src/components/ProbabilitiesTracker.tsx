import React from "react";

interface IProps {
    probabilities: object
}


class ProbabilitiesTracker extends React.Component<IProps> {
    constructor(props: IProps) {
        super(props);
    }

    render() {
        var probs = this.props.probabilities
        return (
            <div className="probabilities-container">
                {Object.keys(probs).map((prob, i) => (
                    <div className="probability" style={{ backgroundColor: `rgba(255, 71, 71, ${probs[prob as keyof typeof probs] * 0.6})`}}>
                        <span className="probability-key">{i+1}. {prob} </span>
                        <span className="probability-value">{parseFloat(probs[prob as keyof typeof probs]).toFixed(5)}</span>
                    </div>
                ))}
            </div>
        )
    }

}

export default ProbabilitiesTracker;