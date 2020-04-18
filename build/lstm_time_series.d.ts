import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptSavedLayers, NestedArray, InputTextArray, PredictionOptions, Shape, TensorScriptLSTMModelContext } from './model_interface';
import { BaseNeuralNetwork } from './base_neural_network';
export declare type TimeSeriesShapeContext = {
    settings: TensorScriptOptions;
    getInputShape: (...args: any[]) => Shape;
};
/**
 * Long Short Term Memory Time Series with Tensorflow
 * @class LSTMTimeSeries
 * @implements {BaseNeuralNetwork}
 */
export declare class LSTMTimeSeries extends BaseNeuralNetwork {
    layers?: TensorScriptSavedLayers;
    /**
     * Creates dataset data
     * @example
     * LSTMTimeSeries.createDataset([ [ 1, ], [ 2, ], [ 3, ], [ 4, ], [ 5, ], [ 6, ], [ 7, ], [ 8, ], [ 9, ], [ 10, ], ], 3) // =>
        //  [
        //    [
        //      [ [ 1 ], [ 2 ], [ 3 ] ],
        //      [ [ 2 ], [ 3 ], [ 4 ] ],
        //      [ [ 3 ], [ 4 ], [ 5 ] ],
        //      [ [ 4 ], [ 5 ], [ 6 ] ],
        //      [ [ 5 ], [ 6 ], [ 7 ] ],
        //      [ [ 6 ], [ 7 ], [ 8 ] ],
        //   ], //x_matrix
        //   [ [ 4 ], [ 5 ], [ 6 ], [ 7 ], [ 8 ], [ 9 ] ] //y_matrix
        // ]
     * @param {Array<Array<number>} dataset - array of values
     * @param {Number} look_back - number of values in each feature
     * @return {[Array<Array<number>>,Array<number>]} returns x matrix and y matrix for model trainning
     */
    static createDataset(dataset?: never[], look_back?: number): any[][];
    /**
     * Reshape input to be [samples, time steps, features]
     * @example
     * LSTMTimeSeries.getTimeseriesShape([
        [ [ 1 ], [ 2 ], [ 3 ] ],
        [ [ 2 ], [ 3 ], [ 4 ] ],
        [ [ 3 ], [ 4 ], [ 5 ] ],
        [ [ 4 ], [ 5 ], [ 6 ] ],
        [ [ 5 ], [ 6 ], [ 7 ] ],
        [ [ 6 ], [ 7 ], [ 8 ] ],
      ]) //=> [6, 1, 3,]
     * @param {Array<Array<number>} x_timeseries - dataset array of values
     * @return {Array<Array<number>>} returns proper timeseries forecasting shape
     */
    static getTimeseriesShape(this: TimeSeriesShapeContext, x_timeseries: NestedArray<any> | undefined): Shape;
    /**
     * Returns data for predicting values
     * @param timeseries
     * @param look_back
     */
    static getTimeseriesDataSet(this: TensorScriptLSTMModelContext, timeseries: never[] | undefined, look_back: any): {
        yShape: Shape;
        xShape: Shape;
        y_matrix: any[];
        x_matrix: Matrix | Vector;
    };
    createDataset: (...args: any[]) => NestedArray<number>;
    getTimeseriesShape: (...args: any[]) => any;
    getTimeseriesDataSet: (...args: any[]) => any;
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
    generateLayers(this: TensorScriptLSTMModelContext, x_matrix: Matrix, y_matrix: Matrix, layers: TensorScriptSavedLayers): void;
    train(x_timeseries: any, y_timeseries: any, layers: any, x_test: any, y_test: any): Promise<any>;
    calculate(x_matrix: Matrix | Vector | InputTextArray): any;
    predict(input_matrix: any[], options: PredictionOptions | undefined): Promise<any>;
}
