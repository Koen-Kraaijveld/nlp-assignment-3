import React from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <button>
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

fetch("https://nlp-assignment-3.onrender.com/predict", {
    // mode: 'no-cors',
    method: "POST",
    body: JSON.stringify({"text": "This fruit is red and round"}),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      // 'Origin': 'http://localhost:3000'
    }
})
.then(function(res) { 
  console.log(res.clone().json()) 
})

export default App;
