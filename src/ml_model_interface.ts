import { TensorScriptModelInterface, TensorScriptOptions, TensorScriptProperties, } from './model_interface';
import { getScikit } from './scikitjs_singleton';

/**
 * Deep Learning with Tensorflow
 * @class MachineLearningModelInterface
 * @implements {TensorScriptModelInterface}
 */
export class MachineLearningModelInterface extends TensorScriptModelInterface {
  scikit: any;
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options:TensorScriptOptions = {}, properties:TensorScriptProperties={}) {
    let scikit = getScikit();
    const config = {
      ...options,
    };
    super(config, properties);
    this.scikit = properties.scikit || scikit;
    this.type = 'MachineLearningModelInterface';
    return this;
  }
  explain(){
    return {
      coefficients: this.model.coef,
      intercept: this.model.intercept,
    };
  }
}