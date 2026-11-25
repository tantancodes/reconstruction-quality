// The 3D viewport: cameras as frustums, the coverage cells, meshes, picking and figure export.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const GREY = new THREE.Color('#3a3a36'), RED = new THREE.Color('#d93a2b'), SEL = new THREE.Color('#0b5fd0'), MESH = new THREE.Color('#cfcdc6');

export class View3D {
  constructor(el, { onPick, onHover } = {}) {
    this.el = el; this.onPick = onPick; this.onHover = onHover;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor('#e6e6e3', 1);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.persp = new THREE.PerspectiveCamera(35, 1, 0.005, 100);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -100, 100);
    this.camera = this.persp;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xa8a49c, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 1.3); key.position.set(1, 2, 1.5); this.scene.add(key);
    this.groups = {};
    this.opts = { frustumSize: 0.022, colorOf: () => GREY, layers: {}, threshold: 0.2 };
    this.selected = -1;
    new ResizeObserver(() => this.resize()).observe(el);
    this.ray = new THREE.Raycaster();
    const ndc = e => { const r = el.getBoundingClientRect(); return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); };
    let down;
    el.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    el.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const i = this.pick(ndc(e)); if (i >= 0 || e.button === 0) this.onPick?.(i);
    });
    el.addEventListener('pointermove', e => { if (e.buttons) return; const i = this.pick(ndc(e)); this.onHover?.(i, e); });
    el.addEventListener('pointerleave', () => this.onHover?.(-1));
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h);
    this.persp.aspect = w / Math.max(h, 1); this.persp.updateProjectionMatrix();
    this.updateOrtho(); this.render();
  }
  updateOrtho() {
    const a = this.el.clientWidth / Math.max(this.el.clientHeight, 1), d = this.camera.position.distanceTo(this.controls.target) * Math.tan(THREE.MathUtils.degToRad(17.5));
    Object.assign(this.ortho, { left: -d * a, right: d * a, top: d, bottom: -d }); this.ortho.updateProjectionMatrix();
  }
  render() { if (this.camera === this.ortho) this.updateOrtho(); this.renderer.render(this.scene, this.camera); }

  clear(name) {
    const g = this.groups[name]; if (!g) return;
    this.scene.remove(g); g.traverse(o => { o.geometry?.dispose(); o.material?.dispose?.(); }); delete this.groups[name];
  }
  add(name, obj) { this.clear(name); this.groups[name] = obj; obj.visible = this.layerOn(name); this.scene.add(obj); }
  layerOn(name) { const l = this.opts.layers[name]; return l === undefined ? true : l; }

  setData(sc) {
    this.sc = sc; this.selected = -1;
    for (const k of Object.keys(this.groups)) this.clear(k);
    this.radius = Math.max(...sc.cameras.map(c => Math.hypot(...c.p)), 0.5);
    this.drawCameras(); this.drawShell(); this.drawAxes(); this.setView('iso');
  }

  // ---------------------------------------------------------------- cameras
  drawCameras() {
    const sc = this.sc; if (!sc) return;
    const size = this.opts.frustumSize * Math.max(this.radius, 0.5) * 2, solid = [], dashed = [], solidC = [], dashedC = [], pts = [], ptc = [];
    const up = new THREE.Vector3(0, 1, 0);
    sc.cameras.forEach((c, i) => {
      const p = new THREE.Vector3(...c.p), v = new THREE.Vector3(...c.v).normalize();
      const side = Math.abs(v.dot(up)) > 0.98 ? new THREE.Vector3(1, 0, 0) : up;
      const r = new THREE.Vector3().crossVectors(v, side).normalize().multiplyScalar(size * 0.6);
      const u = new THREE.Vector3().crossVectors(r, v).normalize().multiplyScalar(size * 0.45);
      const ctr = p.clone().addScaledVector(v, size);
      const k = [ctr.clone().add(r).add(u), ctr.clone().sub(r).add(u), ctr.clone().sub(r).sub(u), ctr.clone().add(r).sub(u)];
      const col = i === this.selected ? SEL : this.opts.colorOf(c, i);
      const [P, C] = c.refOnly ? [dashed, dashedC] : [solid, solidC];
      const seg = (a, b) => { P.push(a.x, a.y, a.z, b.x, b.y, b.z); C.push(col.r, col.g, col.b, col.r, col.g, col.b); };
      k.forEach(q => seg(p, q)); for (let j = 0; j < 4; j++) seg(k[j], k[(j + 1) % 4]);
      seg(k[0].clone().lerp(k[1], 0.5), k[0].clone().lerp(k[1], 0.5).addScaledVector(u.clone().normalize(), size * 0.2));   // tick marks image "up" (roll is arbitrary)
      pts.push(p.x, p.y, p.z); col.toArray(ptc, ptc.length);
    });
    const g = new THREE.Group();
    const lines = (pos, cols, dash) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      const mat = dash ? new THREE.LineDashedMaterial({ vertexColors: true, dashSize: size * 0.15, gapSize: size * 0.12, transparent: true, opacity: 0.8 })
                       : new THREE.LineBasicMaterial({ vertexColors: true });
      const l = new THREE.LineSegments(geo, mat); if (dash) l.computeLineDistances(); return l;
    };
    const fr = new THREE.Group(); fr.name = 'frusta';
    fr.add(lines(solid, solidC, false));
    const refOnly = lines(dashed, dashedC, true); refOnly.name = 'refcams'; refOnly.visible = this.layerOn('refcams'); fr.add(refOnly);
    g.add(fr); fr.visible = this.layerOn('frusta');
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)); pg.setAttribute('color', new THREE.Float32BufferAttribute(ptc, 3));
    this.points = new THREE.Points(pg, new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true }));
    g.add(this.points);
    // optical axes, drawn to the point nearest the origin (the convergence centre)
    const ax = [];
    sc.cameras.forEach(c => { const p = new THREE.Vector3(...c.p), v = new THREE.Vector3(...c.v).normalize(); const t = Math.max(0, -p.dot(v)); ax.push(...c.p, ...p.addScaledVector(v, t).toArray()); });
    const ag = new THREE.BufferGeometry(); ag.setAttribute('position', new THREE.Float32BufferAttribute(ax, 3));
    const rays = new THREE.LineSegments(ag, new THREE.LineBasicMaterial({ color: '#777', transparent: true, opacity: 0.35 })); rays.name = 'rays'; rays.visible = this.layerOn('rays');
    g.add(rays);
    if (this.selected >= 0) {     // nearest neighbour of the selected camera
      const c = sc.cameras[this.selected];
      if (c.nnIndex != null && c.nnIndex >= 0) {
        const used = sc.cameras.filter(x => !x.refOnly), n = used[c.nnIndex];
        if (n) { const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...c.p), new THREE.Vector3(...n.p)]); g.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: SEL }))); }
      }
    }
    this.add('cameras', g);
    this.render();
  }

  pick(ndc) {
    if (!this.points) return -1;
    this.ray.setFromCamera(ndc, this.camera);
    this.ray.params.Points.threshold = this.radius * 0.02;
    const hits = this.ray.intersectObject(this.points);
    return hits.length ? hits[0].index : -1;
  }
  select(i) { this.selected = i; this.drawCameras(); }

  // ---------------------------------------------------------------- coverage cells on a sphere
  drawShell() {
    const sc = this.sc, cov = sc.coverage, g = new THREE.Group();
    const used = sc.cameras.filter(c => !c.refOnly), d = used.map(c => Math.hypot(...c.p)).sort((a, b) => a - b);
    const R = (d[d.length >> 1] || 1) * 1.06;
    const toDisp = (az, el) => { const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el); return [R * Math.cos(e) * Math.cos(a), R * Math.sin(e), -R * Math.cos(e) * Math.sin(a)]; };
    for (let b = 0; b < cov.elEdges.length - 1; b++) for (let a = 0; a < cov.azEdges.length - 1; a++) {
      const pos = [], idx = [], na = 8, ne = 3, inset = 0.6;
      const az0 = cov.azEdges[a] + inset, az1 = cov.azEdges[a + 1] - inset, el0 = cov.elEdges[b] + inset, el1 = cov.elEdges[b + 1] - inset;
      for (let i = 0; i <= ne; i++) for (let j = 0; j <= na; j++) pos.push(...toDisp(az0 + (az1 - az0) * j / na, el0 + (el1 - el0) * i / ne));
      for (let i = 0; i < ne; i++) for (let j = 0; j < na; j++) { const p = i * (na + 1) + j, q = p + na + 1; idx.push(p, q, p + 1, p + 1, q, q + 1); }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(idx);
      const miss = cov.missing[b][a], never = cov.never[b][a];
      const color = miss ? RED : never ? new THREE.Color('#9a9a94') : new THREE.Color('#55554f');
      if (miss || never) g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: miss ? 0.16 : 0.08, side: THREE.DoubleSide, depthWrite: false })));
      g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 25), new THREE.LineBasicMaterial({ color, transparent: true, opacity: miss ? 0.7 : 0.25 })));
    }
    this.add('shell', g);
  }

  drawAxes() {
    const r = this.radius * 1.15, g = new THREE.Group();
    const line = (a, b, c, dashed) => {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
      const l = new THREE.Line(geo, dashed ? new THREE.LineDashedMaterial({ color: c, dashSize: r * 0.04, gapSize: r * 0.03 }) : new THREE.LineBasicMaterial({ color: c }));
      if (dashed) l.computeLineDistances(); return l;
    };
    g.add(line([0, 0, 0], [0, r, 0], '#222'));               // up (ring normal)
    g.add(line([0, 0, 0], [r, 0, 0], '#222', true));         // azimuth 0
    const ring = []; for (let i = 0; i <= 96; i++) { const a = i / 96 * Math.PI * 2; ring.push(new THREE.Vector3(r * 0.8 * Math.cos(a), 0, -r * 0.8 * Math.sin(a))); }
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color: '#999' })));
    this.add('axes', g);
  }

  // ---------------------------------------------------------------- meshes
  async setMeshes(sc) {
    const token = (this.meshToken = {});
    const build = d => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(d.positions, 3));
      if (d.index) geo.setIndex(new THREE.BufferAttribute(d.index, 1));
      geo.computeVertexNormals(); return geo;
    };
    const out = {};
    if (sc.mesh) {
      const d = await sc.mesh(); if (token !== this.meshToken) return out;
      const geo = build(d), n = geo.getAttribute('position').count, col = new Float32Array(n * 3), c = new THREE.Color();
      out.faces = d.index ? d.index.length / 3 : n / 3; out.vertices = n;
      this.errorColors = d.extra ? (() => { const a = new Float32Array(n * 3); for (let i = 0; i < n; i++) c.copy(MESH).lerp(RED, Math.min(1, d.extra[i] / 255 * sc.cutoff / 0.02)).toArray(a, i * 3); return a; })() : null;
      for (let i = 0; i < n; i++) MESH.toArray(col, i * 3);
      this.baseColors = col;
      geo.setAttribute('color', new THREE.BufferAttribute(this.layerOn('error') && this.errorColors ? this.errorColors : col, 3));
      this.meshGeo = geo;
      this.add('mesh', new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
      if (sc.refMesh) {
        const r = await sc.refMesh(); if (token !== this.meshToken) return out;
        const rg = build(r);
        if (d.missing) rg.setAttribute('missing', new THREE.BufferAttribute(d.missing, 1, true));
        this.missingMat = missingMaterial(this.opts.threshold);
        this.add('missing', new THREE.Mesh(rg, this.missingMat));
      }
    }
    this.render(); return out;
  }
  showError(on) { if (this.meshGeo && this.errorColors) { this.meshGeo.setAttribute('color', new THREE.BufferAttribute(on ? this.errorColors : this.baseColors, 3)); this.render(); } }
  setThreshold(fracOfCutoff) { this.opts.threshold = fracOfCutoff; if (this.missingMat) this.missingMat.uniforms.uThresh.value = fracOfCutoff; this.render(); }

  setLayer(name, on) {
    this.opts.layers[name] = on;
    if (name === 'frusta' || name === 'refcams' || name === 'rays') {
      this.groups.cameras?.traverse(o => { if (o.name === name) o.visible = on; });
    } else if (this.groups[name]) this.groups[name].visible = on;
    if (name === 'error') this.showError(on);
    this.render();
  }

  // ---------------------------------------------------------------- views
  setView(kind) {
    const d = this.radius * 3.2, t = new THREE.Vector3();
    const dir = { iso: [1, 0.75, 1.1], top: [0, 1, 0.0001], front: [1, 0.12, 0], side: [0, 0.12, -1] }[kind] || [1, 0.75, 1.1];
    this.camera.up.set(0, 1, 0);
    if (kind === 'top') this.camera.up.set(0, 0, -1);
    this.camera.position.copy(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    this.controls.target.copy(t); this.persp.fov = 35; this.persp.updateProjectionMatrix();
    this.controls.update(); this.render();
  }
  setOrtho(on) {
    const from = this.camera, to = on ? this.ortho : this.persp;
    to.position.copy(from.position); to.up.copy(from.up); this.camera = to; this.controls.object = to;
    this.controls.update(); this.updateOrtho(); this.render();
  }
  lookThrough(i) {
    const c = this.sc.cameras[i]; if (!c) return;
    if (this.camera === this.ortho) this.setOrtho(false);
    const p = new THREE.Vector3(...c.p), v = new THREE.Vector3(...c.v).normalize();
    this.camera.position.copy(p); this.camera.up.set(0, 1, 0);
    this.controls.target.copy(p.clone().addScaledVector(v, Math.max(p.length(), 0.1)));
    this.persp.fov = 50; this.persp.updateProjectionMatrix(); this.controls.update(); this.render();
  }

  async screenshot(scale) {
    const old = this.renderer.getPixelRatio();
    this.renderer.setPixelRatio(scale); this.resize();
    const blob = await new Promise(r => this.renderer.domElement.toBlob(r, 'image/png'));
    this.renderer.setPixelRatio(old); this.resize();
    return blob;
  }
}

function missingMaterial(th) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uThresh: { value: th }, uColor: { value: RED.clone() } },
    vertexShader: `attribute float missing; varying float vM; varying vec3 vN;
      void main() { vM = missing; vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform float uThresh; uniform vec3 uColor; varying float vM; varying vec3 vN;
      void main() { if (vM < uThresh) discard; float shade = 0.78 + 0.22 * abs(normalize(vN).z);
        gl_FragColor = vec4(uColor * shade, 0.75);
        #include <colorspace_fragment>
      }`,
  });
}
