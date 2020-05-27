// import * as tensorflow from '@tensorflow/tfjs';
// import '@tensorflow/tfjs-node';
// import * as tensorflow from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';
;
/******************************************************************************
 * tensorflow.js lambda layer
 * written by twitter.com/benjaminwegener
 * license: MIT
 * @see https://benjamin-wegener.blogspot.com/2020/02/tensorflowjs-lambda-layer.html
 */
export class LambdaLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        if (config.name === undefined) {
            config.name = ((+new Date) * Math.random()).toString(36); //random name from timestamp in case name hasn't been set
        }
        this.name = config.name;
        this.lambdaFunction = config.lambdaFunction;
        if (config.lambdaOutputShape)
            this.lambdaOutputShape = config.lambdaOutputShape;
    }
    call(input, kwargs) {
        // console.log({ input }, 'input[0].shape', input[0].shape)
        // input[0].data().then(inputData=>console.log)
        // console.log('input[0].data()', input[0].data())
        // return input;
        return tf.tidy(() => {
            // return tf.mean(tf.tensor(input),1,true)
            let result = new Array();
            // eval(this.lambdaFunction);
            result = (new Function('input', 'tf', this.lambdaFunction))(input, tf);
            // result = tf.mean(input,1);
            return result;
        });
    }
    computeOutputShape(inputShape) {
        // console.log('computeOutputShape',{inputShape})
        if (this.lambdaOutputShape === undefined) { //if no outputshape provided, try to set as inputshape
            return inputShape[0];
        }
        else {
            return this.lambdaOutputShape;
        }
    }
    getConfig() {
        const config = {
            ...super.getConfig(),
            lambdaFunction: this.lambdaFunction,
            lambdaOutputShape: this.lambdaOutputShape,
        };
        return config;
    }
    static get className() {
        return 'LambdaLayer';
    }
}
/**
 * Base class for tensorscript models
 * @interface TensorScriptModelInterface
 * @property {Object} settings - tensorflow model hyperparameters
 * @property {Object} model - tensorflow model
 * @property {Object} tf - tensorflow / tensorflow-node / tensorflow-node-gpu
 * @property {Function} reshape - static reshape array function
 * @property {Function} getInputShape - static TensorScriptModelInterface
 */
export class TensorScriptModelInterface {
    /**
     * @param {Object} options - tensorflow model hyperparameters
     * @param {Object} customTF - custom, overridale tensorflow / tensorflow-node / tensorflow-node-gpu
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options = {}, properties = {}) {
        // tf.setBackend('cpu');
        this.type = 'ModelInterface';
        /** @type {Object} */
        this.settings = Object.assign({}, options);
        /** @type {Object} */
        this.model = properties.model;
        /** @type {Object} */
        this.tf = properties.tf || tf;
        /** @type {Boolean} */
        this.trained = false;
        this.compiled = false;
        /** @type {Function} */
        this.reshape = TensorScriptModelInterface.reshape;
        /** @type {Function} */
        this.getInputShape = TensorScriptModelInterface.getInputShape;
        if (this.tf && this.tf.serialization && this.tf.serialization.registerClass)
            this.tf.serialization.registerClass(LambdaLayer);
        return this;
    }
    /**
     * Reshapes an array
     * @function
     * @example
     * const array = [ 0, 1, 1, 0, ];
     * const shape = [2,2];
     * TensorScriptModelInterface.reshape(array,shape) // =>
     * [
     *   [ 0, 1, ],
     *   [ 1, 0, ],
     * ];
     * @param {Array<number>} array - input array
     * @param {Array<number>} shape - shape array
     * @return {Array<Array<number>>} returns a matrix with the defined shape
     */
    /* istanbul ignore next */
    static reshape(array, shape) {
        const flatArray = flatten(array);
        function product(arr) {
            return arr.reduce((prev, curr) => prev * curr);
        }
        if (!Array.isArray(array) || !Array.isArray(shape)) {
            throw new TypeError('Array expected');
        }
        if (shape.length === 0) {
            //@ts-ignore
            throw new DimensionError(0, product(size(array)), '!=');
        }
        let newArray;
        let totalSize = 1;
        const rows = shape[0];
        for (let sizeIndex = 0; sizeIndex < shape.length; sizeIndex++) {
            totalSize *= shape[sizeIndex];
        }
        if (flatArray.length !== totalSize) {
            throw new DimensionError(product(shape), 
            //@ts-ignore
            product(size(array)), '!=');
        }
        try {
            newArray = _reshape(flatArray, shape);
        }
        catch (e) {
            if (e instanceof DimensionError) {
                throw new DimensionError(product(shape), 
                //@ts-ignore
                product(size(array)), '!=');
            }
            throw e;
        }
        if (newArray.length !== rows)
            throw new SyntaxError(`specified shape (${shape}) is compatible with input array or length (${array.length})`);
        // console.log({ newArray ,});
        //@ts-ignore
        return newArray;
    }
    /**
     * Returns the shape of an input matrix
     * @function
     * @example
     * const input = [
     *   [ 0, 1, ],
     *   [ 1, 0, ],
     * ];
     * TensorScriptModelInterface.getInputShape(input) // => [2,2]
     * @see {https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array}
     * @param {Array<Array<number>>} matrix - input matrix
     * @return {Array<number>} returns the shape of a matrix (e.g. [2,2])
     */
    //@ts-ignore
    static getInputShape(matrix = []) {
        // static getInputShape(matrix:NestedArray<V>=[]):Shape {
        if (Array.isArray(matrix) === false || !matrix[0] || !matrix[0].length || Array.isArray(matrix[0]) === false)
            throw new TypeError('input must be a matrix');
        const dim = [];
        const x_dimensions = matrix[0].length;
        let vectors = matrix;
        matrix.forEach((vector) => {
            if (vector.length !== x_dimensions)
                throw new SyntaxError('input must have the same length in each row');
        });
        for (;;) {
            dim.push(vectors.length);
            if (Array.isArray(vectors[0])) {
                vectors = vectors[0];
            }
            else {
                break;
            }
        }
        return dim;
    }
    exportConfiguration() {
        return {
            type: this.type,
            settings: this.settings,
            trained: this.trained,
            compiled: this.compiled,
            xShape: this.xShape,
            yShape: this.yShape,
            layers: this.layers,
        };
    }
    importConfiguration(configuration) {
        this.type = configuration.type || this.type;
        this.settings = {
            ...this.settings,
            ...configuration.settings,
        };
        this.trained = configuration.trained || this.trained;
        this.compiled = configuration.compiled || this.compiled;
        this.xShape = configuration.xShape || this.xShape;
        this.yShape = configuration.yShape || this.yShape;
        this.layers = configuration.layers || this.layers;
    }
    train(x_matrix, y_matrix) {
        throw new ReferenceError('train method is not implemented');
    }
    /**
     * Predicts new dependent variables
     * @abstract
     * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
     * @return {{data: Promise}} returns tensorflow prediction
     */
    calculate(matrix) {
        throw new ReferenceError('calculate method is not implemented');
    }
    /**
     * Loads a saved tensoflow / keras model, this is an alias for
     * @param {Object} options - tensorflow load model options
     * @return {Object} tensorflow model
     * @see {@link https://www.tensorflow.org/js/guide/save_load#loading_a_tfmodel}
     */
    async loadModel(options) {
        this.model = await this.tf.loadLayersModel(options);
        this.xShape = this.model.inputs[0].shape;
        this.yShape = this.model.outputs[0].shape;
        this.trained = true;
        return this.model;
    }
    /**
     * saves a tensorflow model, this is an alias for
     * @param {Object} options - tensorflow save model options
     * @return {Object} tensorflow model
     * @see {@link https://www.tensorflow.org/js/guide/save_load#save_a_tfmodel}
     */
    async saveModel(options) {
        const savedStatus = await this.model.save(options);
        return savedStatus;
    }
    /**
     * Returns prediction values from tensorflow model
     * @param {Array<Array<number>>|Array<number>} input_matrix - new test independent variables
     * @param {Boolean} [options.json=true] - return object instead of typed array
     * @param {Boolean} [options.probability=true] - return real values instead of integers
     * @param {Boolean} [options.skip_matrix_check=false] - validate input is a matrix
     * @return {Array<number>|Array<Array<number>>} predicted model values
     */
    // async predict(options?:Matrix|Vector|InputTextArray|PredictionOptions):Promise<Matrix> 
    async predict(input_matrix, options) {
        if (!input_matrix || Array.isArray(input_matrix) === false)
            throw new Error('invalid input matrix');
        const config = {
            json: true,
            probability: true,
            ...options
        };
        const x_matrix = (Array.isArray(input_matrix) || config.skip_matrix_check)
            ? input_matrix
            : [
                input_matrix,
            ];
        return this.calculate(x_matrix)
            .data()
            .then((predictions) => {
            // console.log({ predictions });
            if (config.json === false) {
                return predictions;
            }
            else {
                if (!this.yShape)
                    throw new Error('Model is missing yShape');
                const shape = [x_matrix.length, this.yShape[1],];
                const predictionValues = (config.probability === false) ? Array.from(predictions).map(Math.round) : Array.from(predictions);
                return this.reshape(predictionValues, shape);
            }
        })
            .catch((e) => {
            throw e;
        });
    }
}
/**
 * Calculate the size of a multi dimensional array.
 * This function checks the size of the first entry, it does not validate
 * whether all dimensions match. (use function `validate` for that) (from math.js)
 * @param {Array} x
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @ignore
 * @return {Number[]} size
 */
/* istanbul ignore next */
export function size(x) {
    let s = [];
    while (Array.isArray(x)) {
        s.push(x.length);
        x = x[0];
    }
    return s;
}
/**
 * Iteratively re-shape a multi dimensional array to fit the specified dimensions (from math.js)
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
export function _reshape(array, sizes) {
    // testing if there are enough elements for the requested shape
    var tmpArray = array;
    var tmpArray2;
    // for each dimensions starting by the last one and ignoring the first one
    for (var sizeIndex = sizes.length - 1; sizeIndex > 0; sizeIndex--) {
        var size = sizes[sizeIndex];
        tmpArray2 = [];
        // aggregate the elements of the current tmpArray in elements of the requested size
        var length = tmpArray.length / size;
        for (var i = 0; i < length; i++) {
            tmpArray2.push(tmpArray.slice(i * size, (i + 1) * size));
        }
        // set it as the new tmpArray for the next loop turn or for return
        //@ts-ignore
        tmpArray = tmpArray2;
    }
    //@ts-ignore
    return tmpArray;
}
/**
 * Create a range error with the message:
 *     'Dimension mismatch (<actual size> != <expected size>)' (from math.js)
 * @param {number | number[]} actual        The actual size
 * @param {number | number[]} expected      The expected size
 * @param {string} [relation='!=']          Optional relation between actual
 *                                          and expected size: '!=', '<', etc.
 * @extends RangeError
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
export class DimensionError extends RangeError {
    constructor(actual, expected, relation) {
        /* istanbul ignore next */
        const message = 'Dimension mismatch (' + (Array.isArray(actual) ? ('[' + actual.join(', ') + ']') : actual) + ' ' + ('!=') + ' ' + (Array.isArray(expected) ? ('[' + expected.join(', ') + ']') : expected) + ')';
        super(message);
        this.actual = actual;
        this.expected = expected;
        this.relation = relation;
        // this.stack = (new Error()).stack
        this.message = message;
        this.name = 'DimensionError';
        this.isDimensionError = true;
    }
}
/**
 * Flatten a multi dimensional array, put all elements in a one dimensional
 * array
 * @param {Array} array   A multi dimensional array
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @return {Array}        The flattened array (1 dimensional)
 */
/* istanbul ignore next */
export function flatten(array) {
    /* istanbul ignore next */
    if (!Array.isArray(array)) {
        // if not an array, return as is
        /* istanbul ignore next */
        return array;
    }
    let flat = [];
    /* istanbul ignore next */
    array.forEach(function callback(value) {
        if (Array.isArray(value)) {
            value.forEach(callback); // traverse through sub-arrays recursively
        }
        else {
            flat.push(value);
        }
    });
    return flat;
}
export async function asyncForEach(array, callback) {
    for (let index = 0; index < array.length; index++) {
        await callback(array[index], index, array);
    }
}
