import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import builtins from 'rollup-plugin-node-builtins';
import globals from 'rollup-plugin-node-globals';
import replace from '@rollup/plugin-replace';
import terser from 'rollup-plugin-terser-js';
// import sucrase from '@rollup/plugin-sucrase';
import json from '@rollup/plugin-json';
import pkg from "./package.json";
import alias from '@rollup/plugin-alias';

const name = 'ModelXModel';
const external = [
];
const serverExternal = [
  "@tensorflow-models/universal-sentence-encoder",
  "@tensorflow/tfjs",
  "@tensorflow/tfjs-node",
  "lodash.range",
];
const windowGlobals = {
};

function getOutput({ minify = false, server = false, }) {
  const output = server ?
    [ {
      file: pkg.main,
      format: "cjs",
      exports: "named",
      name,
      sourcemap: true
    },
    {
      file: pkg.esm,
      format: "es",
      exports: "named",
      name,
      sourcemap: true
    }
    ] :
    [ {
      file: pkg.browser,
      format: "umd",
      exports: "named",
      name,
      globals:windowGlobals,
      sourcemap: true
    },
    {
      file: pkg.web,
      format: "iife",
      exports: "named",
      name,
      globals:windowGlobals,
      sourcemap: true
    }
  ];
    if (minify) {
        return output.map(item => {
            const itemFileArray = item.file.split('.');
            itemFileArray.splice(itemFileArray.length - 1, 0, 'min');
            item.file = itemFileArray.join('.');
            item.sourcemap = false;
            return item;
        })
    }
    return output;
}

function getPlugins({
  minify = false,
  browser = false,
  server= false,
}) {
  const plugins = [ ];
  if (browser) {
    plugins.push(
      ...[
        alias({
          // resolve: ['.js', '.ts'],
          entries: {
            '@tensorflow/tfjs-node': '@tensorflow/tfjs',
          }
        })
      ]);
  } 
  
  plugins.push(...[
    // sucrase({
    //   // exclude: ['node_modules/**'],
    //   transforms: [ 'typescript' ]
    // }),
    json(),
    // // external(),
    replace({
      'process.env.NODE_ENV': minify ?
        JSON.stringify('production') : JSON.stringify('development'),
      // 'global.': '(typeof global!=="undefined" ? global : window).'
    }),

    typescript({
      "target": "ES2019",
      noEmitOnError: false,
      declaration: false,
      declarationDir: null,
    }),
    resolve({
      preferBuiltins: true,
    }),
    builtins({}),
    commonjs({
      extensions: [ '.js', '.ts', '.jsx', '.tsx' ]
      // namedExports: {
      //     // 'node_modules/react-is/index.js': ['isValidElementType'],
      //     // 'node_modules/react/index.js': [
      //     //     'Children',
      //     //     'Component',
      //     //     'PropTypes',
      //     //     'createContext',
      //     //     'Fragment',
      //     //     'Suspense',
      //     //     'lazy',
      //     //     'createElement',
      //     //     'useState',
      //     //     'useEffect',
      //     //     'useContext',
      //     //     'useReducer',
      //     //     'useCallback',
      //     //     'useMemo',
      //     //     'useRef',
      //     //     'useImperativeHandle',
      //     //     'useLayoutEffect',
      //     //     'useDebugValue',
      //     //     '__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED',
      //     // ],

      // }
    }), // so Rollup can convert `ms` to an ES module
    globals({
      // react: 'React',
      // 'react-dom': 'ReactDOM'
    }),
  ]);
  if (minify) {
    const minifyPlugins = [

    ].concat(plugins,
      [
        terser({
          sourcemaps: true,
          compress: true,
          mangle: true,
          verbose: true,
        }),
      ]);
    return minifyPlugins;
  }
  return plugins;
}


export default [
  {
    input: "src/index.ts",
    output: getOutput({
      minify: false,
      server: false,
    }),
    external,
    plugins: getPlugins({
      minify: false,
      browser:true,
    }),
  },
  {
    input: "src/index.ts",
    output: getOutput({
      minify: true,
      server: false,
    }),
    external,
    plugins: getPlugins({
      minify: true,
      browser:true,
    }),
  },
  {
    input: "src/index.ts",
    output: getOutput({
      minify: false,
      server: true,
    }),
    external:serverExternal,
    plugins: getPlugins({
      minify: false,
      server: true,
    }),
  },
  {
    input: "src/index.ts",
    output: getOutput({
      minify: true,
      server: true,
    }),
    external:serverExternal,
    plugins: getPlugins({
      minify: true,
      server: true,
    }),
  },
];