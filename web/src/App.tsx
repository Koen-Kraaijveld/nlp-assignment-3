import React from 'react';

import './App.css';
import Main from "./layouts/Main"
import DescriptionInputBar from './components/DescriptionInputBar';
import GameContainer from './components/GameContainer'

function App() {
  return (
    <div className="App">
      <Main>
        <GameContainer />
      </Main>
    </div>
  );
}


export default App;
