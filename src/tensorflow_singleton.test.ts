import { getBackend, setBackend, createModelFitCallback, } from "./tensorflow_singleton";
import * as tf from '@tensorflow/tfjs-node';

describe('tensorflow singleton',()=>{
  describe('getBackend',()=>{
    it('should throw an error if tensorflow has not been set',()=>{
      expect(getBackend.bind(null)).toThrowError(/Looks like you are/);
    });
  });
  describe('setBackend',()=>{
    const tensorflowMockBackend = {};
    it('should set a tf singleton',()=>{
      setBackend(tensorflowMockBackend);
      expect(getBackend()).toBe(tensorflowMockBackend);
    });
  });
  describe('createModelFitCallback',()=>{
    it('should create an array of tensorflow callbacks',()=>{
      setBackend(tf);
      const customCallbacks = {
        onTrainBegin: function(logs: unknown){
          console.log('onTrainBegin', { logs });
        },
        onEpochBegin: function(epoch:number, logs:unknown){
          console.log('onEpochBegin', { epoch, logs });
        },
        onBatchEnd: function(batch:number, logs:unknown){
          console.log('onBatchEnd', { batch, logs });
        },
        onYield: function(epoch:number, batch:number, logs:unknown){
          console.log('onYield', { epoch, batch, logs });
        }
      };
      const tfCallbacks = createModelFitCallback(customCallbacks);
      const onTrainBeginCallback = tfCallbacks[0];
      const tfCallbacksEmpty = createModelFitCallback();
      const customCallbackArray = [1,2];
      expect(onTrainBeginCallback).toBeInstanceOf(tf.CustomCallback)
      expect(tfCallbacks).toHaveLength(Object.keys(customCallbacks).length);
      expect(tfCallbacksEmpty).toHaveLength(0);
      expect(createModelFitCallback(customCallbackArray)).toBe(customCallbackArray)
    });
  })
});