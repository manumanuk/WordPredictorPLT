import React, { Component } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

class Predictor extends Component {
    constructor(props) {
        super(props);
        this.state = {
            value: "",
            newValue: ""
        }
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(event) {
        this.setState({value: event.target.value}, () => {
            if (this.state.value[this.state.value.length-1] == " " && this.state.value[this.state.value.length-2] == " ") {
                this.setState({value: this.state.newValue});
            } else if (this.state.value[this.state.value.length-1] == " ") {
                var xhr = new XMLHttpRequest()
                xhr.addEventListener('load', () => {
                    let newVal = this.state.value + xhr.responseText;
                    this.setState({newValue: newVal});
                });
                let words = this.state.value.split(' ').filter((str) => {if (str != "") return str;});
                if (words.length > 2) {
                    words = words.splice(words.length-2)
                }
                // console.log(words.join(' '));
                xhr.open('GET', 'http://localhost:5000/predict/' + words.join(' ').toLowerCase());
                xhr.send();
                this.setState({newValue: this.state.value + "..."})
            } else {
                this.setState({newValue: this.state.value});
            }
        });
    }
    render() { 
        return (
        <Container className="p-5" fluid style={{backgroundColor: 'black', height: '100vh', color: 'white'}}>
            <Row>
                <Col>
                    <div style={{position: 'relative'}}>
                        <textarea placeholder="Begin Typing! Write a word and press space twice to autofill." value={this.state.value} onChange={this.handleChange} className="display-4 w-100" style={{backgroundColor: 'transparent', color: 'white', boxShadow: 'none', border: 'none', height: '90vh', outline: 'none', position: 'absolute', top: '0', left: '0'}}/>
                        <textarea value={this.state.newValue} className="display-4 w-100" style={{backgroundColor: 'black', color: 'gray', boxShadow: 'none', border: 'none', height: '90vh', outline: 'none', zIndex: -1}}/>
                    </div>
                </Col>
            </Row>
        </Container>
        );
    }
}
 
export default Predictor;