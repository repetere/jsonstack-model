import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import builtins from 'rollup-plugin-node-builtins';
import globals from 'rollup-plugin-node-globals';
import replace from '@rollup/plugin-replace';
import terser from 'rollup-plugin-terser-js';
import json from '@rollup/plugin-json';
import pkg from "./package.json";
import alias from '@rollup/plugin-alias';
import sucrase from '@rollup/plugin-sucrase';

const name = 'ModelXModel';
const external = [
];
const windowGlobals = {
};

function getOutput({ minify = false, server = false, }) {
  const output = [
    {
      file: pkg.browser,
      format: "umd",
      exports: "named",
      name,
      globals: windowGlobals,
      sourcemap: true
    },
    {
      file: pkg.web,
      format: "iife",
      exports: "named",
      name,
      globals: windowGlobals,
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
  plugins.push(
    ...[
      alias({
        // resolve: ['.js', '.ts'],
        entries: {
          '@tensorflow/tfjs-node': '@tensorflow/tfjs',
          'tsne-js': 'tsne-js/build/tsne.min.js',
        }
      })
    ]);
  
  plugins.push(...[
    json(),
    replace({
      'process.env.NODE_ENV': minify ?
      JSON.stringify('production') : JSON.stringify('development'),
    }),
    resolve({
      preferBuiltins: false,
      extensions:['.ts','.js']
    }),
    sucrase({
      transforms:['typescript']
    }),
    builtins({}),
    commonjs({
      extensions: [ '.js', ]
      // namedExports: {
      //     // 'node_modules/react-is/index.js': ['isValidElementType'],
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
];