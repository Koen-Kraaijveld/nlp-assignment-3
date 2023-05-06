import React from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <button onClick={predict}>
        click
      </button>
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

function predict() {
  console.log("Requesting...")
  fetch("https://nlp-assignment-3.onrender.com/predict", {
    // mode: 'no-cors',
    method: "POST",
    body: JSON.stringify({"text": "This mammal hunts prey on the plains of Africa."}),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      // 'Origin': 'http://localhost:3000'
    }
  })
  .then(response => 
    response.json().then(data => ({
        data: data,
        status: response.status
  })))
  .then(res => {
    console.log(res.status, res.data)
  })
  console.log("Received response!")
}


export default App;
