import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createDetailLoader,liveBody} from '../../pss/web/static/live-detail.js';

function harness() {
  const state={scope:{mode:'live',sessionId:'session-a',generation:1},selected:'root',writes:[],errors:[],pending:[]};
  state.loader=createDetailLoader({getScope:()=>state.scope,
    fetchDetail:id=>new Promise((resolve,reject)=>state.pending.push({id,resolve,reject})),
    onDetail:(id,detail)=>state.writes.push({id,detail}),onError:error=>state.errors.push(error.message)});
  return state;
}
test('a late response cannot write/cache into another session with a reused root ID',async()=>{
  const h=harness(),old=h.loader.load('root');
  h.scope={mode:'live',sessionId:'session-b',generation:2};
  const current=h.loader.load('root');
  h.pending[1].resolve({output:'New root'});assert.equal(await current,true);
  h.pending[0].resolve({output:'Old root'});assert.equal(await old,false);
  assert.deepEqual(h.writes,[{id:'root',detail:{output:'New root'}}]);
  assert.equal(h.loader.get('session-a','root'),undefined);
  assert.equal(h.loader.get('session-b','root').output,'New root');
});
test('switching to an example rejects live responses before cache and node writes',async()=>{
  const h=harness(),request=h.loader.load('root');h.scope={mode:'example',sessionId:'quiet-city',generation:2};
  h.pending[0].resolve({output:'Live text'});assert.equal(await request,false);assert.equal(h.writes.length,0);
  assert.equal(h.loader.get('session-a','root'),undefined);
});
test('switching away and back invalidates the old view even for the same session',async()=>{
  const h=harness(),request=h.loader.load('root');h.scope.generation+=2;
  h.pending[0].resolve({output:'Old view'});assert.equal(await request,false);assert.equal(h.writes.length,0);
});
test('selection changes and concurrent full-board loads retain every same-session detail',async()=>{
  const h=harness(),root=h.loader.load('root'),branch=h.loader.load('branch');h.selected='branch';
  h.pending[1].resolve({output:'Branch body'});h.pending[0].resolve({output:'Root body'});
  assert.deepEqual(await Promise.all([root,branch]),[true,true]);assert.equal(h.writes.length,2);
  assert.equal(h.loader.get('session-a','root').output,'Root body');assert.equal(h.loader.get('session-a','branch').output,'Branch body');
});
test('poll and export requests for one branch share a response rather than racing writes',async()=>{
  const h=harness(),poll=h.loader.load('root'),exportRequest=h.loader.load('root');assert.equal(poll,exportRequest);assert.equal(h.pending.length,1);
  h.pending[0].resolve({output:'Shared full text'});assert.equal(await poll,true);assert.equal(h.writes.length,1);
});
test('errors from a departed session do not pollute current status',async()=>{
  const h=harness(),request=h.loader.load('root');h.scope.sessionId='session-b';h.pending[0].reject(Error('Old failure'));
  assert.equal(await request,false);assert.deepEqual(h.errors,[]);
});
test('detail fetching in authored mode never requests the server',async()=>{
  const h=harness();h.scope.mode='example';assert.equal(await h.loader.load('root'),false);assert.equal(h.pending.length,0);
});
test('final output is displayed verbatim and takes precedence over conversation text',()=>{
  const output='  full output\n'+ 'x'.repeat(2000);
  assert.deepEqual(liveBody({output,messages:[{role:'assistant',content:'other'}]}),{body:output,bodySource:'Full live output · model-generated'});
});
test('running branches show all full assistant text with an explicit no-final-output label',()=>{
  const first='  First thought\n',last='<script>literal text</script>'+'x'.repeat(2000);
  const result=liveBody({output:null,messages:[{role:'user',content:'Question'},{role:'assistant',content:first},{role:'tool',content:'tool output'},{role:'assistant',content:last}]});
  assert.equal(result.body,first+'\n\n'+last);assert.match(result.bodySource,/conversation text.*no final output yet/);
});
test('branches with no assistant content explicitly identify their empty state',()=>{
  assert.deepEqual(liveBody({output:'  ',messages:[{role:'user',content:'Input only'}]}),{body:'',bodySource:'No final output or assistant text yet'});
});
