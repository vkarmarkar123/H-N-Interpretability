"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[812],{37796:(e,r,t)=>{t.r(r);t.d(r,{main:()=>K});var o=t(74556);var n=t(95171);var s=t(48435);var l=t(69858);var a=t(48641);var i=t(64331);var c=t(55834);var u=t(28180);var f=t(2542);var A=t(7920);var p=t(38710);var h=t(81554);var y=t(67014);var d=t(77552);var v=t(71821);var b=t(13313);var x=t(23454);var g=t(34802);var j=t(42584);var m=t(54244);var w=t(93814);var C=t(64897);var E=t(94679);var P=t(92121);var O=t(83417);var k=t(83344);var S=t(52109);var _=t(96986);var N=t(63008);var R=t(39721);var J=t(55337);var L=t(32977);var Q=t(95528);var B=t(3268);var I=t(99204);var M=t(38970);var T=t(50168);var U=t(52714);var Y=t(4002);var z=t(57385);var G=t(65540);var V=t(1733);var D=t(30124);async function F(e,r){try{const t=await window._JUPYTERLAB[e].get(r);return t()}catch(t){console.warn(`Failed to create module: package: ${e}; module: ${r}`);throw t}}async function K(){var e=o.PageConfig.getOption("browserTest");if(e.toLowerCase()==="true"){var r=document.createElement("div");r.id="browserTest";document.body.appendChild(r);r.textContent="[]";r.style.display="none";var n=[];var s=false;var l=25e3;var a=function(){if(s){return}s=true;r.className="completed"};window.onerror=function(e,t,o,s,l){n.push(String(l));r.textContent=JSON.stringify(n)};console.error=function(e){n.push(String(e));r.textContent=JSON.stringify(n)}}var i=t(7121).JupyterLab;var c=[];var u=[];var f=[];var A=[];const p=[];const h=[];const y=[];const d=JSON.parse(o.PageConfig.getOption("federated_extensions"));const v=[];d.forEach((e=>{if(e.extension){v.push(e.name);p.push(F(e.name,e.extension))}if(e.mimeExtension){v.push(e.name);h.push(F(e.name,e.mimeExtension))}if(e.style&&!o.PageConfig.Extension.isDisabled(e.name)){y.push(F(e.name,e.style))}}));function*b(e){let r;if(e.hasOwnProperty("__esModule")){r=e.default}else{r=e}let t=Array.isArray(r)?r:[r];for(let n of t){if(o.PageConfig.Extension.isDisabled(n.id)){c.push(n.id);continue}if(o.PageConfig.Extension.isDeferred(n.id)){u.push(n.id);f.push(n.id)}yield n}}const x=[];if(!v.includes("@jupyterlab/javascript-extension")){try{let e=t(37972);for(let r of b(e)){x.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/json-extension")){try{let e=t(44403);for(let r of b(e)){x.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/pdf-extension")){try{let e=t(50635);for(let r of b(e)){x.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/vega5-extension")){try{let e=t(42816);for(let r of b(e)){x.push(r)}}catch(E){console.error(E)}}const g=await Promise.allSettled(h);g.forEach((e=>{if(e.status==="fulfilled"){for(let r of b(e.value)){x.push(r)}}else{console.error(e.reason)}}));if(!v.includes("@jupyterlab/application-extension")){try{let e=t(77636);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/apputils-extension")){try{let e=t(8759);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/cell-toolbar-extension")){try{let e=t(5683);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/celltags-extension")){try{let e=t(34421);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/codemirror-extension")){try{let e=t(98898);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/completer-extension")){try{let e=t(40104);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/console-extension")){try{let e=t(50116);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/csvviewer-extension")){try{let e=t(13763);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/debugger-extension")){try{let e=t(59140);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/docmanager-extension")){try{let e=t(68964);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/documentsearch-extension")){try{let e=t(54434);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/extensionmanager-extension")){try{let e=t(53836);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/filebrowser-extension")){try{let e=t(36325);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/fileeditor-extension")){try{let e=t(6364);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/help-extension")){try{let e=t(61959);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/htmlviewer-extension")){try{let e=t(21464);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/hub-extension")){try{let e=t(28513);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/imageviewer-extension")){try{let e=t(92258);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/inspector-extension")){try{let e=t(55485);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/launcher-extension")){try{let e=t(12543);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/logconsole-extension")){try{let e=t(16688);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/lsp-extension")){try{let e=t(47372);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/mainmenu-extension")){try{let e=t(99886);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/markdownviewer-extension")){try{let e=t(51282);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/markedparser-extension")){try{let e=t(31204);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/mathjax-extension")){try{let e=t(27684);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/metadataform-extension")){try{let e=t(48610);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/notebook-extension")){try{let e=t(48906);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/rendermime-extension")){try{let e=t(81328);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/running-extension")){try{let e=t(97824);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/settingeditor-extension")){try{let e=t(59793);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/shortcuts-extension")){try{let e=t(86151);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/statusbar-extension")){try{let e=t(88637);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/terminal-extension")){try{let e=t(85402);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/theme-dark-extension")){try{let e=t(73638);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/theme-light-extension")){try{let e=t(59340);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/toc-extension")){try{let e=t(21652);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/tooltip-extension")){try{let e=t(42149);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/translation-extension")){try{let e=t(14042);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}if(!v.includes("@jupyterlab/ui-components-extension")){try{let e=t(2731);for(let r of b(e)){A.push(r)}}catch(E){console.error(E)}}const j=await Promise.allSettled(p);j.forEach((e=>{if(e.status==="fulfilled"){for(let r of b(e.value)){A.push(r)}}else{console.error(e.reason)}}));(await Promise.allSettled(y)).filter((({status:e})=>e==="rejected")).forEach((({reason:e})=>{console.error(e)}));const m=new i({mimeExtensions:x,disabled:{matches:c,patterns:o.PageConfig.Extension.disabled.map((function(e){return e.raw}))},deferred:{matches:u,patterns:o.PageConfig.Extension.deferred.map((function(e){return e.raw}))}});A.forEach((function(e){m.registerPluginModule(e)}));m.start({ignorePlugins:f});var w=(o.PageConfig.getOption("exposeAppInBrowser")||"").toLowerCase()==="true";var C=(o.PageConfig.getOption("devMode")||"").toLowerCase()==="true";if(w||C){window.jupyterapp=m}if(e.toLowerCase()==="true"){m.restored.then((function(){a(n)})).catch((function(e){a([`RestoreError: ${e.message}`])}));window.setTimeout((function(){a(n)}),l)}}},7413:e=>{e.exports="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII="}}]);