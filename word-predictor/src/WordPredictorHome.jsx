import React, { Component } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

class Predictor extends Component {
    constructor(props) {
        super(props);
        this.state = {
            value: ""
        }
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(event) {
        this.setState({value: event.target.value}, () => {
            if (this.state.value[this.state.value.length-1] == " " && this.state.value[this.state.value.length-2] == " ") {
                var xhr = new XMLHttpRequest()
                xhr.addEventListener('load', () => {
                    let newVal = this.state.value.slice(0, this.state.value.length-1) + xhr.responseText;
                    this.setState({value: newVal})
                });
                xhr.open('GET', 'https://word-predictor-plt.herokuapp.com/predict/' + this.state.value.toLowerCase());
                xhr.send();
            }
        });
    }
    render() { 
        return (
        <Container className="p-5" fluid style={{backgroundColor: 'black', height: '100vh', color: 'white'}}>
            <Row>
                <Col className="test">
                <textarea placeholder="Begin Typing! Write a word and press space twice to autofill." value={this.state.value} onChange={this.handleChange} className="display-4 w-100" style={{backgroundColor: 'black', color: 'white', boxShadow: 'none', border: 'none', height: '90vh', outline: 'none'}}/>
                </Col>
            </Row>
        </Container>
        );
    }
}
 
export default Predictor;