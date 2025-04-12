import discord
import json
import os
import asyncio
import random
import math
import numpy as np
import copy
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =================================
# Participant & Composite Score Calculation
# =================================

class Participant:
    def __init__(self, name, discord_id, role, hackathon_score, projects_completed,
                 github_contributions, experience_level, problem_solving,
                 innovation_index, availability, wants_to_lead, skill_areas):
        """
        Attributes are self-reported.
        role: e.g., "AI Engineer", "Data Science", "Design", "NeuroScience"
        """
        self.name = name
        self.discord_id = discord_id
        self.role = role
        self.hackathon_score = float(hackathon_score)
        self.projects_completed = int(projects_completed)
        self.github_contributions = int(github_contributions)
        self.experience_level = float(experience_level)
        self.problem_solving = float(problem_solving)
        self.innovation_index = float(innovation_index)
        self.availability = availability
        self.wants_to_lead = wants_to_lead
        self.skill_areas = skill_areas
        self.composite_score = 0.0

    def __repr__(self):
        return f"{self.name} ({self.role}, Composite: {self.composite_score:.2f})"


class CompositeScoreCalculator:
    def __init__(self, participants, weights=None, exponent=3):
        # Default weights (should sum to 1.0)
        if weights is None:
            weights = {
                'hackathon_score': 0.25,
                'projects_completed': 0.2,
                'github_contributions': 0.15,
                'experience_level': 0.1,
                'problem_solving': 0.15,
                'innovation_index': 0.15
            }
        self.participants = participants
        self.weights = weights
        self.exponent = exponent
        self.scaling_factors = self.compute_scaling_factors()

    def compute_scaling_factors(self):
        scaling = {}
        for metric in self.weights.keys():
            values = [getattr(p, metric) for p in self.participants]
            scaling[metric] = {'min': min(values), 'max': max(values)}
        return scaling

    def normalize(self, value, metric):
        min_val = self.scaling_factors[metric]['min']
        max_val = self.scaling_factors[metric]['max']
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def compute_composite_score(self, participant):
        score = 0.0
        for metric, weight in self.weights.items():
            norm_value = self.normalize(getattr(participant, metric), metric)
            score += weight * (norm_value ** self.exponent)
        return score

    def calculate_all_scores(self):
        for p in self.participants:
            p.composite_score = self.compute_composite_score(p)
        return self.participants

# =================================
# Team Formation Algorithm with Updated Logic
# =================================

def initialize_teams(participants, team_size):
    # Do not merge remainder team if smaller than team_size.
    random.shuffle(participants)
    teams = [participants[i:i + team_size] for i in range(0, len(participants), team_size)]
    return teams

def overall_team_avg_variance(teams):
    team_avgs = []
    for team in teams:
        if len(team) > 0:
            team_avgs.append(np.mean([p.composite_score for p in team]))
    return np.var(team_avgs)

def meets_role_requirements(team, role_requirements, leader_required, full_team_size):
    # If the team is not full, skip enforcing role requirements.
    if len(team) < full_team_size:
        return True
    role_counts = {}
    for p in team:
        role_counts[p.role] = role_counts.get(p.role, 0) + 1
    for role, count in role_requirements.items():
        if role_counts.get(role, 0) < count:
            return False
    if leader_required and not any(p.wants_to_lead for p in team):
        return False
    return True

def optimize_teams_sa_anim(teams, role_requirements, leader_required, full_team_size, max_iter=10000, initial_temp=1.0, cooling_rate=0.995):
    current_teams = copy.deepcopy(teams)
    current_obj = overall_team_avg_variance(current_teams)
    best_state = copy.deepcopy(current_teams)
    best_obj = current_obj
    T = initial_temp
    iteration = 0

    yield (copy.deepcopy(current_teams), current_obj, iteration, None, None, None, None, False)

    while T > 1e-4 and iteration < max_iter:
        iteration += 1
        # Choose two random teams to swap members.
        i, j = random.sample(range(len(current_teams)), 2)
        team1 = current_teams[i]
        team2 = current_teams[j]
        if len(team1) == 0 or len(team2) == 0:
            continue
        idx1 = random.randrange(len(team1))
        idx2 = random.randrange(len(team2))
        new_team1 = team1.copy()
        new_team2 = team2.copy()
        new_team1[idx1], new_team2[idx2] = new_team2[idx2], new_team1[idx1]
        if not (meets_role_requirements(new_team1, role_requirements, leader_required, full_team_size) and 
                meets_role_requirements(new_team2, role_requirements, leader_required, full_team_size)):
            T *= cooling_rate
            yield (copy.deepcopy(current_teams), current_obj, iteration, None, None, None, None, False)
            continue
        new_teams = current_teams.copy()
        new_teams[i] = new_team1
        new_teams[j] = new_team2
        new_obj = overall_team_avg_variance(new_teams)
        delta = new_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_teams = new_teams
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_state = copy.deepcopy(current_teams)
            yield (copy.deepcopy(current_teams), current_obj, iteration, i, j, idx1, idx2, False)
        else:
            yield (copy.deepcopy(current_teams), current_obj, iteration, None, None, None, None, False)
        T *= cooling_rate
    yield (copy.deepcopy(best_state), best_obj, iteration, None, None, None, None, True)

def form_teams(participants, team_size, role_requirements, leader_required, max_iter=10000, initial_temp=1.0, cooling_rate=0.995):
    if team_size < sum(role_requirements.values()):
        raise ValueError("Team size must be at least the sum of required roles.")
    teams = initialize_teams(participants, team_size)
    final_state = None
    for state in optimize_teams_sa_anim(teams, role_requirements, leader_required, team_size, max_iter, initial_temp, cooling_rate):
        final_state = state
    return final_state[0] if final_state else teams

# =================================
# Data Storage Functions
# =================================

def load_profiles():
    if os.path.exists("profiles.json"):
        with open("profiles.json", "r") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open("profiles.json", "w") as f:
        json.dump(profiles, f, indent=4)

def load_competition(comp_name):
    filename = f"{comp_name.lower()}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def save_competition(comp_data):
    filename = f"{comp_data['name'].lower()}.json"
    with open(filename, "w") as f:
        json.dump(comp_data, f, indent=4)

# =================================
# Discord Bot Setup & Commands
# =================================

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user}")

# Command: !create-profile
@bot.command(name="create-profile")
async def create_profile(ctx):
    user_id = str(ctx.author.id)
    profiles = load_profiles()
    if user_id in profiles:
        await ctx.send("Profile already exists.")
        return

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    questions = [
        ("Enter your name:", "name"),
        ("Enter your skill areas (comma separated):", "skill_areas"),
        ("Enter your primary role (e.g., Data Science, AI Engineer, Design, NeuroScience):", "role"),
        ("Enter your experience level (1-10):", "experience_level"),
        ("Enter your hackathon score (0-100):", "hackathon_score"),
        ("Enter number of projects completed:", "projects_completed"),
        ("Enter number of GitHub contributions:", "github_contributions"),
        ("Rate your problem solving (1-10):", "problem_solving"),
        ("Rate your innovation (1-10):", "innovation_index"),
        ("Enter your availability (optional):", "availability"),
        ("Do you want to be team lead? (yes/no):", "wants_to_lead")
    ]
    answers = {}
    await ctx.send("Let's create your profile. Please answer the following:")
    for question, key in questions:
        await ctx.send(question)
        try:
            msg = await bot.wait_for("message", timeout=60.0, check=check)
        except asyncio.TimeoutError:
            await ctx.send("Profile creation timed out.")
            return
        answers[key] = msg.content.strip()
    answers["skill_areas"] = [s.strip() for s in answers["skill_areas"].split(",")]
    answers["wants_to_lead"] = answers["wants_to_lead"].lower() in ["yes", "y"]
    profiles[user_id] = answers
    save_profiles(profiles)
    await ctx.send("Profile created successfully.")

# Command: !add-comp <name>
@bot.command(name="add-comp")
@commands.has_permissions(administrator=True)
async def add_comp(ctx, comp_name: str):
    if load_competition(comp_name):
        await ctx.send(f"Competition {comp_name} already exists.")
        return

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    await ctx.send("Creating new competition. Please answer the following:")
    await ctx.send("Enter competition type (e.g., AI Hackathon, Design Hackathon, Case Competition):")
    try:
        comp_type_msg = await bot.wait_for("message", timeout=60.0, check=check)
    except asyncio.TimeoutError:
        await ctx.send("Competition creation timed out.")
        return
    comp_type = comp_type_msg.content.strip()

    await ctx.send("Enter required roles and counts. Format: Role1:Count, Role2:Count (e.g., AI Engineer:1, Data Science:1, Design:1, NeuroScience:1):")
    try:
        roles_msg = await bot.wait_for("message", timeout=60.0, check=check)
    except asyncio.TimeoutError:
        await ctx.send("Competition creation timed out.")
        return
    roles_input = roles_msg.content.strip()
    role_requirements = {}
    try:
        for part in roles_input.split(","):
            role, count = part.split(":")
            role_requirements[role.strip()] = int(count.strip())
    except Exception:
        await ctx.send("Error parsing role requirements. Please follow the format.")
        return

    await ctx.send("Should each team have a team leader? (yes/no):")
    try:
        leader_msg = await bot.wait_for("message", timeout=60.0, check=check)
    except asyncio.TimeoutError:
        await ctx.send("Competition creation timed out.")
        return
    leader_required = leader_msg.content.strip().lower() in ["yes", "y"]

    comp_data = {
        "name": comp_name,
        "type": comp_type,
        "role_requirements": role_requirements,
        "leader_required": leader_required,
        "participants": []
    }
    save_competition(comp_data)
    await ctx.send(f"Competition {comp_name} ({comp_type}) created with role requirements: {role_requirements} and leader_required = {leader_required}")

    # Create a new category with the competition name
    comp_category = discord.utils.get(ctx.guild.categories, name=comp_name)
    if comp_category is None:
        comp_category = await ctx.guild.create_category(comp_name)
    # Create the two default channels: Competition Info and Help.
    try:
        await ctx.guild.create_text_channel("competition-info", category=comp_category)
        await ctx.guild.create_text_channel("help", category=comp_category)
    except Exception as e:
        await ctx.send(f"Error creating default channels: {e}")

# Command: !join-comp <comp_name>
@bot.command(name="join-comp")
async def join_comp(ctx, comp_name: str):
    user_id = str(ctx.author.id)
    profiles = load_profiles()
    if user_id not in profiles:
        await ctx.send("You need to create a profile using !create-profile first.")
        return
    comp_data = load_competition(comp_name)
    if not comp_data:
        await ctx.send(f"Competition {comp_name} does not exist.")
        return
    if user_id in comp_data["participants"]:
        await ctx.send("You have already joined this competition.")
        return
    comp_data["participants"].append(user_id)
    save_competition(comp_data)
    await ctx.send(f"You have joined {comp_name}.")

# Command: !list-participants <comp_name>
@bot.command(name="list-participants")
async def list_participants(ctx, comp_name: str):
    comp_data = load_competition(comp_name)
    if not comp_data or not comp_data["participants"]:
        await ctx.send("No participants in this competition.")
        return
    profiles = load_profiles()
    message = f"Participants in {comp_name}:\n"
    for uid in comp_data["participants"]:
        profile = profiles.get(uid, {})
        name = profile.get("name", "Unknown")
        message += f"- {name}\n"
    await ctx.send(message)

# Command: !make-teams <comp_name> <team_size>
@bot.command(name="make-teams")
async def make_teams(ctx, comp_name: str, team_size: int):
    comp_data = load_competition(comp_name)
    if not comp_data:
        await ctx.send(f"Competition {comp_name} does not exist.")
        return
    profiles = load_profiles()
    if not comp_data["participants"]:
        await ctx.send("No participants in this competition.")
        return

    participants_list = []
    for uid in comp_data["participants"]:
        profile = profiles.get(uid)
        if not profile:
            continue
        try:
            participant = Participant(
                name = profile.get("name"),
                discord_id = uid,
                role = profile.get("role"),
                hackathon_score = float(profile.get("hackathon_score", 0)),
                projects_completed = int(profile.get("projects_completed", 0)),
                github_contributions = int(profile.get("github_contributions", 0)),
                experience_level = float(profile.get("experience_level", 0)),
                problem_solving = float(profile.get("problem_solving", 0)),
                innovation_index = float(profile.get("innovation_index", 0)),
                availability = profile.get("availability", ""),
                wants_to_lead = profile.get("wants_to_lead", False),
                skill_areas = profile.get("skill_areas", [])
            )
        except Exception as e:
            await ctx.send(f"Error processing profile for user {uid}: {e}")
            continue
        participants_list.append(participant)

    if len(participants_list) < team_size:
        await ctx.send("Not enough participants to form a team.")
        return

    try:
        calc = CompositeScoreCalculator(participants_list)
        calc.calculate_all_scores()
    except Exception as e:
        await ctx.send(f"Error calculating scores: {e}")
        return

    role_requirements = comp_data.get("role_requirements", {})
    leader_required = comp_data.get("leader_required", False)
    if team_size < sum(role_requirements.values()):
        await ctx.send("The team size is too small for the required role composition.")
        return

    try:
        teams = form_teams(participants_list, team_size, role_requirements, leader_required)
    except Exception as e:
        await ctx.send(f"Team formation failed: {e}")
        return

    message = f"**{comp_name} ({comp_data['type']}) Teams**\n"
    for idx, team in enumerate(teams, 1):
        message += f"\n**Team {idx}**\n"
        for p in team:
            member = ctx.guild.get_member(int(p.discord_id))
            mention = member.mention if member else p.name
            message += f"- {mention} ({p.role})\n"
        if leader_required:
            leaders = [p.name for p in team if p.wants_to_lead]
            leader_text = leaders[0] if leaders else "No volunteer"
            message += f"→ Team Lead: {leader_text}\n"
    await ctx.send(message)

    # Auto-create private team channels inside the competition category.
    if ctx.guild.me.guild_permissions.manage_channels:
        comp_category = discord.utils.get(ctx.guild.categories, name=comp_name)
        if comp_category is None:
            comp_category = await ctx.guild.create_category(comp_name)
        for idx, team in enumerate(teams, 1):
            channel_name = f"{comp_name.lower()}-team-{idx}"
            overwrites = { ctx.guild.default_role: discord.PermissionOverwrite(read_messages=False) }
            for p in team:
                member = ctx.guild.get_member(int(p.discord_id))
                if member:
                    overwrites[member] = discord.PermissionOverwrite(read_messages=True)
            try:
                channel = await ctx.guild.create_text_channel(channel_name, overwrites=overwrites, category=comp_category)
                team_mentions = []
                for p in team:
                    member = ctx.guild.get_member(int(p.discord_id))
                    if member:
                        team_mentions.append(member.mention)
                await channel.send(f"Welcome {' '.join(team_mentions)}! This is your private team channel for {comp_name}.")
            except Exception as e:
                print(f"Failed to create channel {channel_name}: {e}")

# =================================
# Run the Bot
# =================================

bot.run(os.getenv('DISCORD_TOKEN'))
