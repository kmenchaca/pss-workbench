import {examples,visibleNodes,exportBoard} from './examples.js';
import {element,renderGraph} from './graph.js';
import {createDetailLoader,liveBody} from './live-detail.js';
const $=id=>document.getElementById(id);
const state={board:examples[1],mode:'example',selected:'gift',pruned:new Set(),compared:new Set(),config:null,live:null,generation:0,busy:false};
let exportText='',polling=false;
const details=createDetailLoader({
  getScope:()=>({mode:state.mode,sessionId:state.board.id,generation:state.generation}),
  fetchDetail:id=>api(`/api/context/${encodeURIComponent(id)}`),
  onDetail:(id,detail)=>{const node=state.board.nodes.find(n=>n.id===id);if(node){Object.assign(node,liveBody(detail));node.messages=detail.messages??[];node.full=true;}if(state.selected===id)renderDetail();},
  onError:error=>notice(error.message),
});
function notice(text){$('notice').textContent=text;}
function visible(){return visibleNodes(state.board.nodes,state.pruned);}
function button(text,fn,className=''){const b=element('button',className,text);b.addEventListener('click',fn);return b;}
function open(id){$(id).showModal();}
async function api(path,body){
  const response=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','x-pss-token':state.config?.csrf_token??''}:{},body:body?JSON.stringify(body):undefined});
  let data;try{data=await response.json();}catch{throw Error('The local server returned an unreadable response.');}
  if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:`Request failed (${response.status}). Check the local server.`);return data;
}
function loadExample(example){state.generation++;state.mode='example';state.board=example;state.selected=example.id==='first-contact'?'gift':example.nodes[1].id;state.pruned.clear();state.compared.clear();notice('');$('boards-dialog').close();render();}
function render(){
  const nodes=visible();if(!nodes.some(n=>n.id===state.selected))state.selected=nodes[0]?.id;
  $('title').textContent=state.board.title;$('prompt').textContent=state.board.prompt;$('prompt').tabIndex=0;
  $('mode').textContent=state.mode==='example'?'Authored demo · no runtime model call':`Live provider run · ${state.live?.outcome??'starting'}`;
  $('mode').classList.toggle('live-mode',state.mode==='live');
  $('branch-count').textContent=`${nodes.length} thoughts · ${nodes.filter(n=>n.parent).length} branches`;
  $('restore').disabled=!state.pruned.size;$('restore').hidden=state.mode==='live';$('reset').hidden=state.mode==='live';
  $('compare-count').textContent=state.compared.size;$('compare').disabled=state.compared.size<2;
  $('live-controls').hidden=state.mode!=='live'||!state.live?.exploration_active;
  $('board-hint').textContent=state.mode==='example'?'Select a card to read the full thought. Check two or three branches to compare.':`${state.live?.request_count??0} requests · ${state.live?.output_tokens??0} output tokens · ${state.live?.message||'Select a branch for its full output.'}`;
  document.querySelectorAll('[data-example]').forEach(b=>b.classList.toggle('active',state.mode==='example'&&b.dataset.example===state.board.id));
  const selected=nodes.find(n=>n.id===state.selected);
  $('breadcrumb').textContent=selected?.parent?selected.title:'THE QUESTION';
  const canFollow=!!selected?.next&&nodes.some(n=>n.id===selected.next);
  $('follow').disabled=state.mode!=='example'||!canFollow;
  $('set-aside').disabled=state.mode!=='example'||!selected?.parent;
  $('read-branch').disabled=!selected;
  $('follow-hint').textContent=state.mode==='live'?'Live controls are in the full branch view':canFollow?'Opens an authored continuation · no new generation':selected?.next?'Continuation set aside · restore to follow it':'End of this authored path · explore a sibling';
  renderGraph($('graph'),nodes,state.selected,state.compared,id=>select(id),id=>toggleCompare(id));renderDetail();fitGraph();
  // The fitted transform establishes screen-space question clearance. Reflow
  // once at that scale so peripheral cards respect the actual prompt bounds.
  renderGraph($('graph'),nodes,state.selected,state.compared,id=>select(id),id=>toggleCompare(id));fitGraph();
}
async function select(id){state.selected=id;render();if(state.mode==='live')await fetchDetail(id);}
function fetchDetail(id){return details.load(id);}
function toggleCompare(id){
  if(state.compared.has(id))state.compared.delete(id);else if(state.compared.size<3)state.compared.add(id);else notice('Compare up to three branches. Uncheck one to choose another.');render();
}
function renderDetail(){
  const node=state.board.nodes.find(n=>n.id===state.selected),pane=$('detail');pane.replaceChildren();if(!node){pane.append(element('p','detail-empty','The graph will appear as the first context is created.'));return;}
  $('detail-index').textContent=`${String(state.board.nodes.indexOf(node)+1).padStart(2,'0')} / ${String(state.board.nodes.length).padStart(2,'0')}`;
  pane.append(element('p','detail-lens',node.lens||node.status||'LIVE OUTPUT'),element('h2','detail-title',node.title));
  if(node.parent){const parent=state.board.nodes.find(n=>n.id===node.parent);pane.append(button(`↖ From ${parent?.title??'parent branch'}`,()=>select(node.parent),'parent-link'));}
  pane.append(element('p','detail-provenance',state.mode==='example'?'Authored example · not model-generated':node.full?node.bodySource:'Live preview only · fetching full output'));
  pane.append(element('div','detail-body',node.body||'This branch has no output yet.'));
  if(state.mode==='live'&&node.terminationReason)pane.append(element('p','detail-provenance',`Termination: ${node.terminationReason}`));
  if(state.mode==='live'&&node.messages?.length){const history=element('details','message-history');history.append(element('summary','',`Inspect full conversation (${node.messages.length} messages)`));node.messages.forEach(message=>{history.append(element('h3','',message.role),element('p','detail-body',message.content));});pane.append(history);}
  if(node.tradeoff){const box=element('div','tradeoff');box.append(element('h3','','THE TRADEOFF'),element('p','',node.tradeoff));pane.append(box);}
  const actions=element('div','detail-actions');
  if(state.mode==='example'){
    if(node.next&&visible().some(n=>n.id===node.next))actions.append(button('Follow authored continuation →',()=>select(node.next),'primary'));
    if(node.parent)actions.append(button('Set this branch aside',setAside));
    actions.append(element('p','action-note',node.next?'Opens an existing authored path. No new text is generated.':'Select a sibling on the board to explore another direction.'));
  }else if(state.live?.exploration_active&&node.status==='running'){
    actions.append(button('Prune live branch',()=>command('kill',node.id),'danger'),button('Request a branch',()=>command('promote',node.id)),button('Add instruction',()=>{$('inject-text').value='';open('inject-dialog');}));
    actions.append(element('p','action-note','Live commands affect the active run and can trigger additional provider work within its limits.'));
  }else actions.append(element('p','action-note','Finished branches cannot restart. Start a new exploration to continue a thought.'));
  pane.append(actions);
}
async function compare(){
  const generation=state.generation,ids=[...state.compared];
  if(state.mode==='live'){const loaded=await Promise.all(ids.map(fetchDetail));if(generation!==state.generation)return;if(loaded.some(ok=>!ok)){notice('Could not load full output for comparison. Try again.');return;}}
  $('comparison').replaceChildren();$('comparison-provenance').textContent=state.mode==='example'?'Curated alternatives · compare these authored directions, not model scores.':'Live model outputs · compare the arguments yourself; no quality score is assigned.';
  ids.forEach(id=>{const n=state.board.nodes.find(n=>n.id===id);if(!n)return;const column=element('section','compare-column');column.append(element('p','detail-lens',n.lens||n.status),element('h3','',n.title));if(state.mode==='live')column.append(element('p','detail-provenance',n.bodySource));column.append(element('p','compare-body',n.body||'No final output or assistant text is available for this branch yet.'));if(n.tradeoff)column.append(element('h4','','Tradeoff'),element('p','',n.tradeoff));$('comparison').append(column);});open('compare-dialog');
}
async function config(){try{state.config=await api('/api/config');$('connection').textContent=state.config.live_enabled?'Live provider configured':'Examples ready · live optional';}catch(error){$('connection').textContent='Examples ready · server unavailable';state.config=null;}renderSetup();}
function renderSetup(){const c=state.config;$('setup-status').textContent=c?.live_enabled?`Ready: ${c.provider} / ${c.model}`:c?'Live calls are disabled. Curated examples work without a key.':'Local server unavailable. Start the Python workbench to enable live configuration.';$('provider-notice').textContent=c?.notice??'Examples are local authored content and never call a model.';$('setup-instructions').textContent=c&&!c.live_enabled?c.setup:'';$('setup-instructions').hidden=!!c?.live_enabled;$('run-limits').textContent=c?`Limits: ${c.max_calls??18} requests · ${c.max_output_tokens_per_call??768} output tokens per call · ${c.max_seconds??300} seconds. These are not a strict dollar-cost cap.`:'';$('start-live').disabled=!c?.live_enabled||state.busy||!!state.live?.exploration_active;}
async function command(type,target,payload){try{const result=await api('/api/command',{type,target,payload});notice(result.message||'Command accepted.');await poll();}catch(error){notice(error.message);}}
function liveBoard(data){
  const contexts=Array.isArray(data.tree?.contexts)?data.tree.contexts:Object.values(data.tree?.contexts??{});
  const nodes=contexts.map((c,i)=>{const detail=details.get(data.session_id,c.id);return{id:c.id,parent:c.parent_id,title:c.branch_reason||(!c.parent_id?'Initial exploration':`Branch ${i+1}`),lens:!c.parent_id?'LIVE ROOT':`LIVE · ${c.status}`,summary:(c.output||c.last_message||'Waiting for output…').slice(0,110),...(detail?liveBody(detail):{body:c.output??'',bodySource:'Live preview only'}),messages:detail?.messages??[],full:!!detail,status:c.status,terminationReason:c.termination_reason,tradeoff:''};});
  return{id:data.session_id,title:'Live exploration',prompt:data.prompt||'Provider exploration',nodes};
}
async function poll(){if(polling)return;polling=true;const generation=state.generation;try{const data=await api('/api/state');if(generation!==state.generation)return;const sessionChanged=state.live?.session_id!==data.session_id;state.live=data;if(sessionChanged)details.clear();$('live-board').hidden=!data.session_id;if(state.mode==='live'){if(state.board.id!==data.session_id){state.generation++;state.compared.clear();state.selected=null;}state.board=liveBoard(data);render();if(state.selected)await fetchDetail(state.selected);}renderSetup();}catch(error){if(generation===state.generation&&state.mode==='live'){notice(`Connection interrupted: ${error.message} Displayed output may be stale.`);$('connection').textContent='Server disconnected';}}finally{polling=false;}}
function setAside(){const node=state.board.nodes.find(n=>n.id===state.selected);if(state.mode!=='example'||!node?.parent)return;state.pruned.add(node.id);state.compared=new Set([...state.compared].filter(id=>visible().some(n=>n.id===id)));state.selected=visible().find(n=>n.parent===node.parent)?.id||node.parent;notice('Branch and its continuations set aside. Restore pruned brings them back.');render();}
let zoom=1;
function fitGraph(){const viewport=document.querySelector('.canvas-scroll'),graph=$('graph'),stage=document.querySelector('.graph-stage');if(window.matchMedia('(max-width: 650px)').matches){graph.style.transform='';graph.style.left='';stage.style.width='';stage.style.height='';return;}const graphHeight=parseFloat(graph.style.height)||735;const scale=Math.min(viewport.clientWidth/1488,viewport.clientHeight/graphHeight,1.4)*zoom;graph.style.transform=`scale(${scale})`;graph.style.left=`${Math.max(0,(viewport.clientWidth-1488*scale)/2)}px`;stage.style.width=`${1488*scale}px`;stage.style.height=`${graphHeight*scale}px`;}
function setZoom(value){zoom=Math.max(.65,Math.min(1.4,value));$('zoom').value=Math.round(zoom*100);$('zoom-value').textContent=`${Math.round(zoom*100)}%`;render();}
new ResizeObserver(()=>render()).observe(document.querySelector('.canvas-scroll'));
$('zoom').addEventListener('input',e=>setZoom(Number(e.target.value)/100));$('zoom-out').addEventListener('click',()=>setZoom(zoom-.1));$('zoom-in').addEventListener('click',()=>setZoom(zoom+.1));$('fit').addEventListener('click',()=>{setZoom(1);document.querySelector('.canvas-scroll').scrollTo(0,0);});
$('root-nav').addEventListener('click',()=>{const root=visible().find(n=>!n.parent);if(root)select(root.id);});
$('boards-open').addEventListener('click',()=>open('boards-dialog'));
$('read-branch').addEventListener('click',()=>open('detail-dialog'));
$('follow').addEventListener('click',()=>{const node=visible().find(n=>n.id===state.selected);if(state.mode==='example'&&node?.next&&visible().some(n=>n.id===node.next))select(node.next);});
$('set-aside').addEventListener('click',setAside);
examples.forEach((ex,i)=>{const b=button('',()=>loadExample(ex));b.dataset.example=ex.id;b.append(element('span','example-num',`0${i+1}`),element('strong','',ex.short),element('small','',ex.subtitle));$('examples').append(b);});
document.querySelectorAll('[data-close]').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
$('reset').addEventListener('click',()=>loadExample(state.board));$('restore').addEventListener('click',()=>{state.pruned.clear();notice('All authored branches restored.');render();});$('compare').addEventListener('click',compare);
$('live-setup').addEventListener('click',()=>{$('boards-dialog').close();open('setup-dialog');config();});
$('live-board').addEventListener('click',()=>{if(!state.live?.session_id)return;$('boards-dialog').close();state.generation++;state.mode='live';state.board=liveBoard(state.live);state.pruned.clear();state.compared.clear();state.selected=state.board.nodes[0]?.id;render();if(state.selected)fetchDetail(state.selected);});
document.querySelectorAll('[data-command]').forEach(b=>b.addEventListener('click',()=>command(b.dataset.command)));
$('run-form').addEventListener('submit',async event=>{event.preventDefault();if(state.busy)return;state.busy=true;renderSetup();$('run-error').textContent='';try{const prompt=$('live-prompt').value;const started=await api('/api/run',{prompt,preset:null,config_overrides:{max_contexts:Number($('context-limit').value)}});state.generation++;state.mode='live';state.pruned.clear();state.compared.clear();details.clear();state.selected=null;state.live={session_id:started.session_id,prompt,exploration_active:true,outcome:'starting'};state.board=liveBoard(state.live);render();$('setup-dialog').close();notice('Live provider run started.');await poll();}catch(error){$('run-error').textContent=error.message;}finally{state.busy=false;renderSetup();}});
$('inject-form').addEventListener('submit',async event=>{event.preventDefault();try{await api('/api/command',{type:'inject',target:state.selected,payload:$('inject-text').value});$('inject-dialog').close();notice('Instruction sent to the running branch.');}catch(error){$('inject-error').textContent=error.message;}});
$('export').addEventListener('click',async()=>{const generation=state.generation;if(state.mode==='live'){notice('Loading full branch content for export…');const loaded=await Promise.all(visible().map(n=>fetchDetail(n.id)));if(generation!==state.generation)return;if(loaded.some(ok=>!ok)){notice('Export stopped: full content was unavailable. Try again after reconnecting.');return;}}exportText=exportBoard(state.board,visible(),state.mode);$('export-preview').value=exportText;open('export-dialog');});
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([exportText],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`pss-${state.board.id||'exploration'}.md`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
render();config().then(poll);setInterval(()=>{if(state.config)poll();},1800);
