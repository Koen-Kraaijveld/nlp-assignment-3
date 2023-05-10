import React from 'react';
import fs from 'fs'

import { ImSpinner2 } from "react-icons/im"
import { AiOutlineEnter } from "react-icons/ai"

import ProbabilitiesTracker from './ProbabilitiesTracker';
import categories from "../data/categories_100.json";

interface IProps {
}

interface IState {
    description: string,
    isRequesting: boolean,
    currentCategory: string,
    possibleCategories: Array<string>,
    score: number,
    currentPrediction: object,
    isShowingPopup: boolean,
    popupText: string
}

class GameContainer extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props);
        this.state = {
            description: "",
            isRequesting: false,
            currentCategory: "",
            possibleCategories: categories,
            score: 0,
            currentPrediction: {},
            isShowingPopup: false,
            popupText: "Correct!"
        }
        this.handleSubmit = this.handleSubmit.bind(this);
        
    }

    componentDidMount(): void {
        alert("Please note that the first request sent might take upwards of 2 minutes. Due to limited resources, the AI model used in this app is hosted on a free web service. Therefore, the API might take some time to respond. Apologies for the inconvenience.")
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
        if (this.state.description.includes(this.state.currentCategory)) {
            this.setState({isShowingPopup: true})
            this.setState({popupText: "Your answer cannot contain the given word!"})
            setTimeout(() => {
                this.setState({isShowingPopup: false})
            }, 2000);
        } else {
            this.setState({isShowingPopup: false})
            this.predict(this.state.description)
        }
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
            const highestProbLabel = Object.keys(res.data)[0]

            if (highestProbLabel == this.state.currentCategory) {
                this.setState({score: this.state.score + 1})
                this.setRandomCategory()
                this.setState({isShowingPopup: true})
                this.setState({popupText: "Correct!"})
                setTimeout(() => {
                    this.setState({isShowingPopup: false})
                }, 2000);
            } else {
                this.setState({score: 0})
                this.setState({isShowingPopup: true})
                this.setState({popupText: "Game over!"})
                setTimeout(() => {
                    this.setState({isShowingPopup: false})
                }, 2000);
            }
            this.setState({isRequesting: false})
            this.setState({currentPrediction: res.data})
            console.log(res.data)
        })
    }
    
    render() {
        return (
            <div>
                <div className="description-input-wrapper">
                    <div className="description-input-container">
                        {
                        !this.state.isShowingPopup && 
                        <div>
                            <div className="score-container">
                                <p className="game-info">Score:</p>
                                <p>{this.state.score}</p>
                            </div>
                            <div className="category-container">
                                <p className="game-info">Describe the word:</p>
                                <p>{this.state.currentCategory}</p>
                            </div>
                        </div>
                        }

                        {
                        this.state.isShowingPopup &&
                        <div className="popup-text-container"> 
                            {this.state.popupText}
                        </div>
                        }
                        
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
                <ProbabilitiesTracker probabilities={this.state.currentPrediction}/>
            </div>
            
        )
    }
    
}

export default GameContainer;