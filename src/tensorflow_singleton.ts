import { CustomCallback } from "@tensorflow/tfjs-layers";

let tf: any = null;
export type TrainingProgressUpdate = {
  completion_percentage?: number;
  loss?: number;
  epoch?: number;
  logs: {
    loss: number
  };
  status?: string;
  batch?:number;
  defaultLog?: boolean;
}

export type TrainingProgressCallback =(...args:any[])=> void;

export type CustomCallbackFunctions = {
  [index: string]: TrainingProgressCallback
}

export function setBackend(tfInput: any) {
  tf = tfInput;
}

export function getBackend() {
  if (tf === null) {
    throw Error(`
============================
Howdy 👋👋. Looks like you are running @jsonstack/model but you haven't set a Tensorflow backend. 
To do so, simply import (or require) your tensorflow library, and call setBackend like so,

import * as tf from '@tensorflow/tfjs';
import * as jsm from '@jsonstack/model';
jsm.setBackend(tf);

That will let @jsonstack/model know you wish to use a tensorflow library to perform your calculations.
============================
    `);
  }
  return tf;
}

export function createModelFitCallback(callbackFunctions?: CustomCallbackFunctions|any[]){
  const tf = getBackend();

  if(Array.isArray(callbackFunctions)){ 
    return callbackFunctions;
  } else if(callbackFunctions){
    return Object.keys(callbackFunctions).map((callbackFunctionName:string)=>{
      return new tf.CustomCallback({ [callbackFunctionName]: callbackFunctions[callbackFunctionName]})
    });
  } else return [];
}