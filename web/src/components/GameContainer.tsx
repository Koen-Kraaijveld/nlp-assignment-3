import React from 'react';

import DescriptionInputBar from './DescriptionInputBar';

interface IProps {
}

interface IState {
}

class GameContainer extends React.Component<IProps, IState> {
    constructor(props: IProps) {
        super(props)
        this.state = {
        }
    }

    render() {
        return (
            <div>
                <DescriptionInputBar />
            </div>
        )
    }

}

export default GameContainer