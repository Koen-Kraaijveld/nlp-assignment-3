import React from 'react';

import { ImSpinner2 } from "react-icons/im"
import { AiOutlineEnter } from "react-icons/ai"

interface IProps {
}

interface IState {
    description: string,
    isRequesting: boolean,
    currentCategory: string,
    possibleCategories: Array<string>,
    score: number,
    currentPrediction: object
}

class DescriptionInputBar extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            description: "",
            isRequesting: false,
            currentCategory: "",
            possibleCategories: ["apple", "angel", "jail"],
            score: 0,
            currentPrediction: {}
        }
        this.handleSubmit = this.handleSubmit.bind(this);
        
    }

    componentDidMount(): void {
        this.setState({
            currentCategory: this.state.possibleCategories[Math.floor(Math.random() * this.state.possibleCategories.length)]
        })
    }

    setRandomCategory() {
        var categories = this.state.possibleCategories
        var randomCategory = categories[Math.floor(Math.random() * categories.length)]
        this.setState({currentCategory: randomCategory})
    }

    handleSubmit(e: any) {
        e.preventDefault()
        this.predict(this.state.description)
    }
    
    predict(description: string) {
        this.setState({isRequesting: true})
        fetch("https://nlp-assignment-3.onrender.com/predict", {
          method: "POST",
          body: JSON.stringify({"text": description}),
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          }
        })
        .then(response => 
            response.json().then(data => ({
              data: data,
              status: response.status
        })))
        .then(res => {
            console.log(res.status, res.data)
            const sortable = Object.fromEntries(
                Object.entries(res.data).sort(([,a],[,b]) => a-b)
            );
        

            if (res.data.label == this.state.currentCategory) {
                this.setState({score: this.state.score + 1})
                this.setRandomCategory()
            }
            this.setState({isRequesting: false})
        })
    }
    
    render() {
        return (
            <div className="description-input-wrapper">
                <div className="description-input-container">
                    <div className="score-container">
                        <p>score</p>
                        <p>{this.state.score}</p>
                    </div>
                    <div className="category-container">
                        {this.state.currentCategory}
                    </div>
                    <div className="input-form-container">
                        <form onSubmit={this.handleSubmit}>
                            <input type="text" onChange={(e) => this.setState({description: e.target.value})} disabled={this.state.isRequesting}/>
                            <button id="btn-submit" onClick={this.handleSubmit} disabled={this.state.isRequesting}>
                                {this.state.isRequesting ? 
                                <ImSpinner2 className="spinning" size={30}/> :
                                <AiOutlineEnter size={30}/>}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        )
    }
    
}

export default DescriptionInputBar;