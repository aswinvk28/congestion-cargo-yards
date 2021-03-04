import React from 'react';
import Area from './Area';

export default class Screen extends ReadableByteStreamController.Component {

    render() {
        return (
            <div className="screen">
                <Area description="TECHNICAL" />
                <Area description="SAFETY" />
                <Area description="PROCESS" />
                <Area description="OPERATIONS" />
            </div>
        )
    }

}