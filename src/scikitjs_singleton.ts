let scikit: any = null;

export function setScikit(scikitInput: any) {
  scikit = scikitInput;
}

export function getScikit() {
  if (scikit === null) {
    throw Error(`
============================
Howdy 👋👋. Looks like you are running @jsonstack/model but you haven't set a ScikitJS backend. 
To do so, simply import (or require) your scikit library, and call setScikit like so,

import * as tf from '@tensorflow/tfjs';
import * as scikit from 'scikitjs';
import * as jsm from '@jsonstack/model';
jsm.setBackend(tf);
jsm.setScikit(scikit);

That will let @jsonstack/model know you wish to use a scikitjs library to perform your calculations.
============================
    `);
  }
  return scikit;
}
