import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptLayers, PredictionOptions, DenseLayer, Corpus } from './model_interface';
import { BaseNeuralNetwork } from './base_neural_network';
/**
 * use a corpus to generate features from an embedding layer with Tensorflow
 * @class FeatureEmbedding
 * @implements {BaseNeuralNetwork}
 */
export declare class FeatureEmbedding extends BaseNeuralNetwork {
    layers?: TensorScriptLayers;
    featureToId?: any;
    IdToFeature?: any;
    featureIds?: any;
    numberOfFeatures?: any;
    static getFeatureDataSet(this: any, { inputMatrixFeatures, PAD, }: {
        inputMatrixFeatures: Corpus;
        PAD?: string;
    }): Promise<{
        featureToId: {
            [x: string]: number;
        };
        IdToFeature: {
            0: any;
        };
        featureIds: number[][];
        numberOfFeatures: number;
    }>;
    static getMergedArray(base?: Vector, merger?: Vector, append?: boolean): any[];
    /**
     */
    static getContextPairs(this: any, { inputMatrix, numberOfFeatures, window_size, tf, }: {
        inputMatrix: Matrix;
        numberOfFeatures: number;
        window_size?: number;
        tf?: any;
    }): Promise<{
        context_length: number;
        emptyXVector: any;
        emptyYVector: any;
        x: Matrix;
        y: Matrix;
    }>;
    getMergedArray: typeof FeatureEmbedding.getMergedArray;
    getFeatureDataSet: typeof FeatureEmbedding.getFeatureDataSet;
    getContextPairs: typeof FeatureEmbedding.getContextPairs;
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
    /**
     * Adds dense layers to tensorflow classification model
     * @override
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @param {Array<Object>} layers - model dense layer parameters
     */
    generateLayers(this: FeatureEmbedding, x_matrix: Matrix, y_matrix: Matrix, layers?: TensorScriptLayers): void;
    train(x_matrix: Matrix | Corpus, y_matrix: Matrix, layers?: DenseLayer[]): Promise<any>;
    calculate(): Promise<any>;
    predict(options?: PredictionOptions): Promise<any>;
    /**
     * Converts matrix of layer weights into labeled features
     * @example
  const weights = [
    [1.5,1,4,1.6,3.5],
    [4.3,3.2,5.5,6.5]
  ]
  FeatureEmbeddingInstance.labelWeights(weights) //=>
  weights = {
    car:[1.5,1,4,1.6,3.5],
    boat:[4.3,3.2,5.5,6.5]
  }
     */
    labelWeights(weights: Matrix): {
        [index: string]: Vector;
    };
    /**
     * Uses tSNE to reduce dimensionality of features
     * @example
  const weights = [
    [1.5,1,4,1.6,3.5],
    [4.3,3.2,5.5,6.5]
  ]
  FeatureEmbeddingInstance.reduceWeights(weights) //=>
  [
    [1,2],
    [2,3],
  ]
     */
    reduceWeights(weights: Matrix, options?: any): Promise<any>;
    /**
     * Uses either cosineProximity or Eucledian distance to rank similarity
    @example
    //weights = [ [1,2,3,], [1,2,2], [0,-1,3] ]
    //labeledWeights = [ {car:[1,2,3,],tesla:[1,2,2],boat:[0,-1,3]}]
    FeatureEmbeddingInstance.findSimilarFeatures(weights,{features:['car'], limit:2,}) //=>
    {
      car:[
      {
        comparedFeature: 'tesla',
        proximity: -0.5087087154388428,
        distance: 0.03015853278338909
      },
      {
        comparedFeature: 'boat',
        proximity: -0.3032159209251404,
        distance: 0.036241017282009125
      },
      ]
    }
     */
    findSimilarFeatures(weights: Matrix, options?: SimilarFeatureOptions): Promise<SimilarFeatures>;
}
export declare type LabeledWeights = {
    [index: string]: Vector;
};
export declare enum SimilarityMetric {
    DISTANCE = "distance",
    PROXIMITY = "proximity"
}
export declare type SimilarFeatureOptions = {
    features?: string[];
    limit?: number;
    labeledWeights?: LabeledWeights;
    metric?: SimilarityMetric;
};
export declare type SimilarFeatures = {
    [index: string]: SimilarFeature[];
};
export declare type SimilarFeature = {
    comparedFeature: string;
    distance: number;
    proximity: number;
};
