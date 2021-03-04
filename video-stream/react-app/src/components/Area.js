import React from 'react';
import NotifIcon from './NotifIcon';

export default class Area extends React.Component {

    color = '#bc0000'

    constructor(props) {
        super(props)
    }
    
    render() {
        <div className="area" style={{backgroundColor: this.color}}>
            {this.props.description}
            <NotifIcon />
        </div>
    }

}