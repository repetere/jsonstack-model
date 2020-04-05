import { TensorScriptOptions, TensorScriptProperties, Matrix, TensorScriptLayers } from './model_interface';
import { BaseNeuralNetwork } from './base_neural_network';
/**
 * Logistic Regression Classification with Tensorflow
 * @class LogisticRegression
 * @implements {BaseNeuralNetwork}
 */
export declare class LogisticRegression extends BaseNeuralNetwork {
    /**
     * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
    /**
     * Adds dense layers to tensorflow classification model
     * @override
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @param {Array<Object>} layers - model dense layer parameters
     * @param {Array<Array<number>>} x_test - validation data independent variables
     * @param {Array<Array<number>>} y_test - validation data dependent variables
     */
    generateLayers(x_matrix: Matrix, y_matrix: Matrix, layers: TensorScriptLayers, x_test: Matrix, y_test: Matrix): void;
}
