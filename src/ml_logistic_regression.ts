import { TensorScriptOptions, TensorScriptProperties, Matrix, TensorScriptLayers, InputTextArray, PredictionOptions, Vector, Calculation, } from './model_interface';
import { MachineLearningModelInterface, } from './ml_model_interface';
import { createModelFitCallback } from './tensorflow_singleton';

/**
 * Machine Learning Linear Regression with Tensorflow
 * @class MachineLearningLogisticRegression
 * @implements {MachineLearningModelInterface}
 */
export class MachineLearningLogisticRegression extends MachineLearningModelInterface {
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object,layerPreference:String}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options:TensorScriptOptions = {}, properties?:TensorScriptProperties) {
    const config = {
      layers: [],
      layerPreference:'deep',
      compile: {
      },
      fit: {
        callbacks:[]
      },
      ...options,
    };
    super(config, properties);
    
    if(options?.fit?.callbacks) config.fit.callbacks =[        
      this.tf.callbacks.earlyStopping({ monitor: 'loss', patience: 50 })
    ].concat(createModelFitCallback(options?.fit?.callbacks));

    this.type = 'MachineLearningLogisticRegression';
    this.model = new this.scikit.LogisticRegression({
      modelFitOptions: config.fit
    });

    return this;
  }
  /**
   * Asynchronously trains tensorflow model
   * @override
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @return {Object} returns trained tensorflow model 
   */
  async train(x_matrix:Matrix, y_matrix:Matrix) {
    const trainable_y_matrix = y_matrix.map(y=>y[0])
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    // const xs = this.tf.tensor(x_matrix, xShape);
    // const ys = this.tf.tensor(trainable_y_matrix, yShape);
    this.xShape = xShape;
    this.yShape = yShape;
    await this.model.fit(x_matrix, trainable_y_matrix, this.settings.fit);
    this.trained = true;
    this.compiled = true;
    return this.model;
  }
  /**
   * Predicts new dependent variables
   * @override
   * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
   * @param {Object} options - model prediction options
   * @return {{data: Promise}} returns tensorflow prediction 
   */
   calculate(input_matrix:Matrix|Vector, options?:PredictionOptions) {
    //TODO: @dcrmls you could the shape here
    if (!input_matrix || Array.isArray(input_matrix)===false) throw new Error('invalid input matrix');
    const predictionInput = (Array.isArray(input_matrix[ 0 ]))
      ? input_matrix
      : [
        input_matrix,
      ];
    const predictionTensor = this.tf.tensor(predictionInput);
    const prediction = this.model.predict(predictionTensor, options);
    predictionTensor.dispose();
    return prediction;
  }
}