import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {examples,visibleNodes,exportBoard} from '../../pss/web/static/examples.js';

test('all three examples have complete original content and valid graph links',()=>{
  assert.equal(examples.length,3);
  for(const ex of examples){
    assert.equal(ex.nodes.length,7);
    const ids=new Set(ex.nodes.map(n=>n.id));assert.equal(ids.size,ex.nodes.length);
    assert.equal(ex.nodes.filter(n=>!n.parent).length,1);
    for(const n of ex.nodes){assert.ok(n.body.length>120);if(n.parent)assert.ok(ids.has(n.parent));if(n.next)assert.ok(ids.has(n.next));}
  }
});
test('pruning excludes an entire authored subtree and restore recovers it',()=>{
  const ex=examples[0],pruned=new Set(['routes']);
  assert.deepEqual(visibleNodes(ex.nodes,pruned).map(n=>n.id),['root','rooms','rhythm','room-test','rhythm-test']);
  pruned.clear();assert.equal(visibleNodes(ex.nodes,pruned).length,7);
});
test('cycles do not hang traversal',()=>{assert.deepEqual(visibleNodes([{id:'a',parent:'b'},{id:'b',parent:'a'}],new Set()),[]);});
test('export contains each visible full body, source question and honest provenance',()=>{
  const ex=examples[0],visible=visibleNodes(ex.nodes,new Set(['routes'])),text=exportBoard(ex,visible,'example');
  assert.ok(text.includes('not a recorded run'));assert.ok(text.includes(ex.prompt));
  for(const n of visible)assert.ok(text.includes(n.body));
  assert.ok(!text.includes(ex.nodes.find(n=>n.id==='routes').body));
});
test('live export preserves literal output and full conversation without truncation',()=>{
  const content='<script>unsafe()</script>\n'+'full output '.repeat(400);
  const text=exportBoard({title:'Live',prompt:'Question'},[{id:'a',parent:null,title:'Branch',body:content,messages:[{role:'user',content:'full input'}]}],'live');
  assert.ok(text.includes(content));assert.ok(text.includes('full input'));assert.ok(text.includes('Live provider output'));assert.ok(!text.includes('no model calls'));
});
test('model content uses safe DOM text insertion and all network mutations use the CSRF boundary',async()=>{
  const app=await readFile(new URL('../../pss/web/static/app.js',import.meta.url),'utf8');
  const graph=await readFile(new URL('../../pss/web/static/graph.js',import.meta.url),'utf8');
  assert.ok(!app.includes('innerHTML'));assert.ok(!graph.includes('innerHTML'));assert.ok(graph.includes('textContent = text'));
  assert.ok(app.includes("'x-pss-token'"));assert.ok(app.includes('encodeURIComponent(id)'));
});
