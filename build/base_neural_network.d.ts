import { TensorScriptModelInterface, TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptLayers, TensorScriptSavedLayers, PredictionOptions } from './model_interface';
/**
 * Deep Learning with Tensorflow
 * @class BaseNeuralNetwork
 * @implements {TensorScriptModelInterface}
 */
export declare class BaseNeuralNetwork extends TensorScriptModelInterface {
    /**
     * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
    /**
     * Adds dense layers to tensorflow model
     * @abstract
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @param {Array<Object>} layers - model dense layer parameters
     */
    generateLayers(x_matrix: Matrix, y_matrix: Matrix, layers?: TensorScriptLayers | TensorScriptSavedLayers, x_test?: Matrix, y_test?: Matrix): void;
    /**
     * Asynchronously trains tensorflow model
     * @override
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @param {Array<Object>} layers - array of model dense layer parameters
     * @param {Array<Array<number>>} x_text - validation data independent variables
     * @param {Array<Array<number>>} y_text - validation data dependent variables
     * @return {Object} returns trained tensorflow model
     */
    train(x_matrix: Matrix, y_matrix: Matrix, layers: TensorScriptLayers, x_test: Matrix, y_test: Matrix): Promise<any>;
    /**
     * Predicts new dependent variables
     * @override
     * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
     * @param {Object} options - model prediction options
     * @return {{data: Promise}} returns tensorflow prediction
     */
    calculate(input_matrix: Matrix | Vector, options?: PredictionOptions): any;
}
